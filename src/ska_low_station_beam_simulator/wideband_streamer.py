"""
LEGACY wideband time-domain generation + FFT channelization streamer.

STATUS: superseded by direct_synthesis.DirectSynthesisStreamer for
tone+noise configs (what StationSimulatorDevice actually uses by default —
see simulator.py). Kept only as the fallback for source_cfgs containing
kind='pulsed' — pulsed sources are explicitly PARKED for direct synthesis
(no derived closed-form per-channel representation yet; see
direct_synthesis.py's module docstring). Per the handoff doc's own scaling
analysis, this path is very unlikely to be viable at the full 448-channel
band regardless of further optimization, since its cost scales worse than
linearly with channel count (the FFT) against a fixed per-tick budget.

Deliberately NOT merged into the same module as direct_synthesis.py — the
two approaches share no code (different generation model, different delay
handling, different output path to "channelized" samples) and the FFT
approach is expected to be dropped entirely once direct synthesis covers
pulsed sources too. Keeping them side by side as separate files makes it
easy to delete this one later without touching the other.

UNFIXED BUG #12 (still applies here, NOT fixed in this module): station
(receiver) noise gets delay-corrected identically to the sky signal in
StationStreamer.generate_next_tick (both `shared` and `noise` are summed
into `raw[pol]` and go through the SAME ring-buffer/coarse-fine-delay/
channelization pipeline). That's physically wrong — receiver noise
originates locally per station, after any signal-path delay would apply.
DirectSynthesisStreamer fixes this for free (noise never enters a delay
pipeline there). Only fix it here too if this legacy path ends up seeing
real use beyond pulsed-only fallback duty — not worth the restructuring
otherwise.

WHAT'S TRUSTWORTHY (unchanged from the earliest version, still
real/tested logic):
    - Stateless, seekable signal sources (tone/noise/pulsed).
    - Coarse+fine delay tracking from a time-varying polynomial.
    - Wideband channelization producing all channels from one FFT.
    - Producer/sender decoupling via a bounded queue (see common.py).
    - Deterministic sim_time derivation.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import scipy.fft
from numba import njit, prange

from ska_low_station_beam_simulator.common import (
    BASE_FREQ_HZ,
    BLOCK_DURATION_S,
    CHANNEL_WIDTH_HZ,
    DelayPolynomial,
    NUM_CHANNELS,
    StationConfig,
    fetch_delay_model_from_cbf,
    log,
)

# Historically capped to 1 to prevent numpy/BLAS's own thread pool from
# oversubscribing against an even older ThreadPoolExecutor-chunked
# generation design. That reason no longer applies — generation now runs
# via numba's own separate threading layer, not numpy/BLAS at all. This
# cap may now be needlessly limiting numpy's FFT (channelization) to a
# single thread, right as cross-hardware benchmarking confirmed
# channelization is the dominant remaining cost (~70-74% of the tick).
# Test on real hardware by setting these env vars BEFORE importing this
# module (they're only defaults here, not hardcoded overrides), e.g.:
#   OMP_NUM_THREADS=2 python3 -m ska_low_station_beam_simulator.benchmark
# Try a few small values (2, 4) rather than unbounding entirely — with
# 2 channelization tasks (V, H) potentially each spawning their own BLAS
# thread pool, an uncapped value risks reintroducing the SAME kind of
# oversubscription problem this cap was originally added to prevent,
# just relocated to channelization instead of generation. Measure with
# benchmark.py's phase-breakdown section specifically, since it isolates
# channelization time cleanly from the rest of the tick.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


# ============================================================
# CONFIG — backend-specific to this wideband+FFT approach. NUM_CHANNELS,
# CHANNEL_WIDTH_HZ, BASE_FREQ_HZ, BLOCK_DURATION_S live in common.py since
# both backends need them.
# ============================================================

# For critical sampling with no oversampling, per-channel output rate
# equals CHANNEL_WIDTH_HZ, which requires step == FFT_LEN (no overlap-save
# window reuse). OVERLAP > 0 here means each channel's OUTPUT sample rate
# is HIGHER than CHANNEL_WIDTH_HZ (an oversampled channelizer) — whether
# that matches your ICD's actual SPS channelization scheme is NOT
# verified here. If your ICD is critically sampled, set OVERLAP = 0.
# NOTE: common.py's BLOCK_DURATION_S and ScanRunner's channel_output_rate
# assume critical sampling (OVERLAP=0, FFT_LEN=NUM_CHANNELS) — changing
# either constant below without updating those would desync heap framing.
FFT_LEN = NUM_CHANNELS
OVERLAP = 0
SAMPLE_RATE_HZ = NUM_CHANNELS * CHANNEL_WIDTH_HZ  # wideband input rate, ~75 MHz here

# Thread pool sizing for the channelization wave ONLY (task-level
# concurrency between V and H, not FFT-internal parallelism — see
# FFT_WORKERS below for that). Generation runs via numba/prange,
# controlled separately by numba.set_num_threads(). Channelization only
# ever submits 2 concurrent tasks (V, H) per tick, so this pool doesn't
# need sweeping — 2 is enough by construction, more just sits idle.
NUM_WORKER_THREADS = 2

# FFT worker count for scipy.fft (numpy.fft has NO multi-threading
# support at all — this is why we switched). NOT the same knob as
# numba's thread count or NUM_WORKER_THREADS (the channelization task
# executor) — this controls parallelism WITHIN one scipy.fft.fft() call.
#
# EMPIRICALLY 1 IS BEST, confirmed on real hardware (pinned and unpinned
# EPYC): FFT_WORKERS > 1 made things WORSE, not better — likely because
# scipy.fft's own internal thread pool contends with numba's thread pool
# (already resident from the generation phase, up to NUMBA_NUM_THREADS
# threads) rather than adding genuine independent parallelism. The
# scipy.fft swap itself still appears to be a modest real win over
# numpy.fft even at workers=1 (not yet cleanly isolated from other
# changes — worth a dedicated A/B if you need to confirm that
# specifically) — keep the swap, just don't parallelize it further.
FFT_WORKERS = 1


def split_coarse_fine(tau_s: float, sample_rate: float) -> tuple[int, float]:
    tau_samples = tau_s * sample_rate
    coarse = int(round(tau_samples))
    fine_s = (tau_samples - coarse) / sample_rate
    return coarse, fine_s


# ============================================================
# SIGNAL SOURCES — stateless, seekable
# ============================================================


class Source:
    def generate(self, t_start: float, n: int, sample_rate: float) -> np.ndarray:
        raise NotImplementedError


class ToneSource(Source):
    """Periodic CW tone via a precomputed lookup table (nearest-neighbor
    indexing) instead of calling exp() over the full sample range every
    invocation — exp() was the dominant remaining cost after threading
    divided (but didn't eliminate) it: ~17.5ms for one un-chunked tick's
    worth of samples, measured earlier in this conversation.

    IMPORTANT — unlike the ring-buffer and FFT-vectorization fixes
    elsewhere, this trades a small, BOUNDED approximation for speed; it
    is not numerically identical to direct exp(). Worst-case phase error
    is pi/table_size radians (~2.4e-5 rad at the default table_size=65536)
    — several orders of magnitude below both the downstream 8-bit
    quantization step and any delay-tracking tolerance discussed earlier,
    but it IS an approximation. If you ever need bit-exact tone
    generation (e.g. a high-precision analytic reference for
    cross-checking), revert to direct exp() or raise table_size (at
    proportionally reduced speed benefit).

    Still fully stateless/seekable: the table is precomputed once at
    construction and never mutated, so generate(t_start, n, sample_rate)
    remains a pure function of its inputs — the resumability/multi-pod
    safety property from earlier is preserved.

    Table dtype is complex64, not complex128: the table's own
    quantization error already dominates float32's ~1e-7 relative
    precision loss by several orders of magnitude, so complex128 here
    would cost memory bandwidth for no real benefit.
    """

    def __init__(self, freq_hz: float, amplitude: float = 1.0, table_size: int = 65536):
        self.freq_hz = freq_hz
        self.amplitude = amplitude
        self.table_size = table_size
        angles = 2 * np.pi * np.arange(table_size) / table_size
        self._table = (amplitude * np.exp(1j * angles)).astype(np.complex64)

    def generate(self, t_start: float, n: int, sample_rate: float) -> np.ndarray:
        sample_idx = t_start * sample_rate + np.arange(n)
        cycle_frac = np.mod(sample_idx * (self.freq_hz / sample_rate), 1.0)
        # round to NEAREST table entry (not truncate) — truncation gives
        # up to a full bin (2*pi/table_size) of error; rounding halves
        # that to the pi/table_size bound documented above. Caught by
        # the correctness check below when the first version used
        # .astype(int64) directly on cycle_frac*table_size (floor).
        table_idx = (
            np.round(cycle_frac * self.table_size).astype(np.int64) % self.table_size
        )
        return self._table[
            table_idx
        ]  # complex64 — upcasts automatically on += into complex128 accumulators


class NoiseSource(Source):
    """Counter-based (Philox) RNG keyed by absolute sample index. Verify
    Philox's counter-seeking semantics against your numpy version.

    Generates in float32 (via standard_normal's native dtype support,
    scaled afterward) rather than float64 — a real precision reduction,
    not free like the ring-buffer/FFT fixes. float32 Gaussian noise has
    ~7 significant decimal digits, utterly swamping the downstream 8-bit
    quantization step; flagged explicitly since, unlike the tone
    lookup table, there's no table-quantization error already dominating
    it — this reduction stands on its own and is worth being aware of
    if noise statistics at high precision ever matter to a test.
    """

    def __init__(self, std: float, seed: int):
        self.std = std
        self.seed = seed

    def generate(self, t_start: float, n: int, sample_rate: float) -> np.ndarray:
        start_sample = int(round(t_start * sample_rate))
        bg = np.random.Philox(key=self.seed, counter=start_sample)
        rng = np.random.Generator(bg)
        scale = self.std / np.sqrt(2)
        real = rng.standard_normal(n, dtype=np.float32) * scale
        imag = rng.standard_normal(n, dtype=np.float32) * scale
        return real + 1j * imag


class PulsedSource(Source):
    def __init__(self, period_s: float, width_s: float, amplitude: float = 1.0):
        self.period_s = period_s
        self.width_s = width_s
        self.amplitude = amplitude

    def generate(self, t_start: float, n: int, sample_rate: float) -> np.ndarray:
        t = t_start + np.arange(n) / sample_rate
        on = np.mod(t, self.period_s) < self.width_s
        return np.where(on, self.amplitude, 0.0).astype(np.complex128)


def build_source(cfg: dict) -> Source:
    kind = cfg["kind"]
    if kind == "tone":
        return ToneSource(cfg["freq_hz"], cfg.get("amplitude", 1.0))
    if kind == "noise":
        return NoiseSource(cfg["std"], cfg["seed"])
    if kind == "pulsed":
        return PulsedSource(cfg["period_s"], cfg["width_s"], cfg.get("amplitude", 1.0))
    raise ValueError(f"unknown source kind: {kind}")


# ============================================================
# NUMBA-ACCELERATED GENERATION — replaces an even older
# ThreadPoolExecutor-based chunked generation wave. Confirmed by direct
# benchmarking that ThreadPoolExecutor's plateau at ~4-8 workers,
# consistent across three different CPU architectures, was per-task
# Python/GIL overhead, not compute or memory bandwidth — numba's prange
# showed near-linear scaling well past that point on the same hardware.
#
# SCOPING DECISION: this fast path covers 'tone' and 'pulsed' source
# kinds, plus the per-pol station (receiver) noise — i.e. exactly what
# build_source() above supports AND what every StartScan/benchmark call
# in this codebase actually configures. A source_cfgs entry with
# kind='noise' (an injected SHARED noise-like source, distinct from
# per-pol receiver noise) is NOT covered by the numba path — it isn't
# exercised anywhere in this codebase currently, so accelerating it
# speculatively felt like the wrong tradeoff. It still works via the
# original NoiseSource class as a fallback (see StationStreamer below),
# just without the speedup. If you start using it, say so and this can
# be extended the same way tone/pulsed were.
#
# IMPORTANT — Philox is NOT supported in numba nopython mode (confirmed
# by direct test). Noise here (both the fast-path station noise AND the
# tone lookup table math) uses different, from-scratch implementations:
# a splitmix64-style hash + Box-Muller for noise. This means:
#   - station (receiver) noise is numerically DIFFERENT from the old
#     Philox-based NoiseSource sequence — same statistical properties
#     (verified: mean~0, std matches request), NOT the same values.
#   - if you have test fixtures built against old station-noise output,
#     they will not match and need regenerating.
#
# direct_synthesis.py has its own, independent copy of a similar
# splitmix64+Box-Muller kernel (kept separate deliberately — see this
# module's docstring on why the two backends don't share code).
# ============================================================

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def build_tone_table_unit(table_size: int = 65536) -> np.ndarray:
    """Unit-amplitude lookup table, shared across all configured tones —
    each tone applies its own amplitude as a separate multiply, so one
    table serves any number of simultaneously configured tones."""
    angles = 2 * np.pi * np.arange(table_size) / table_size
    return np.exp(1j * angles).astype(np.complex64)


@njit(cache=True)
def _tone_sample_unit(table, table_size, freq_hz, sample_rate, sample_idx_abs):
    cycle_frac = (sample_idx_abs * (freq_hz / sample_rate)) % 1.0
    idx = int(round(cycle_frac * table_size)) % table_size
    return table[idx]


@njit(cache=True)
def _splitmix64_hash(seed, index):
    x = (np.uint64(seed) ^ (np.uint64(index) * GOLDEN)) & MASK64
    x = (x + GOLDEN) & MASK64
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    z = z ^ (z >> np.uint64(31))
    return z


@njit(cache=True)
def _uniform_from_hash(h):
    return np.float64(h >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _gaussian_pair(seed, sample_idx_abs):
    u1 = _uniform_from_hash(_splitmix64_hash(seed, 2 * sample_idx_abs))
    u2 = _uniform_from_hash(_splitmix64_hash(seed, 2 * sample_idx_abs + 1))
    u1 = max(u1, 1e-300)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    return r * np.cos(theta), r * np.sin(theta)


@njit(parallel=True, cache=True)
def generate_shared_signal(
    tone_table,
    table_size,
    tone_freqs,
    tone_amps,
    pulse_periods,
    pulse_widths,
    pulse_amps,
    sample_rate,
    t_start_rel,
    n,
):
    """Sum of all configured tones + pulses, over n samples. Computed
    ONCE per tick (not once per pol) — this content doesn't depend on
    polarisation at all (per-pol delay is applied downstream, after
    channelization). An earlier ThreadPoolExecutor-based version
    recomputed this identically twice per tick, once per pol — a genuine
    redundancy fixed here as a side effect, not the main point of the
    numba port."""
    out_real = np.zeros(n, dtype=np.float64)
    out_imag = np.zeros(n, dtype=np.float64)
    n_tones = tone_freqs.shape[0]
    n_pulses = pulse_periods.shape[0]
    start_sample_abs = t_start_rel * sample_rate

    for i in prange(n):
        sample_idx_abs = start_sample_abs + i
        t = t_start_rel + i / sample_rate
        acc_real = 0.0
        acc_imag = 0.0
        for k in range(n_tones):
            val = _tone_sample_unit(
                tone_table, table_size, tone_freqs[k], sample_rate, sample_idx_abs
            )
            acc_real += val.real * tone_amps[k]
            acc_imag += val.imag * tone_amps[k]
        for k in range(n_pulses):
            phase_in_period = t % pulse_periods[k]
            if phase_in_period < pulse_widths[k]:
                acc_real += pulse_amps[k]
                # imag left at 0 for pulses — matches original PulsedSource's
                # real-valued on/off behaviour, cast to complex.
        out_real[i] = acc_real
        out_imag[i] = acc_imag

    return out_real + 1j * out_imag


@njit(parallel=True, cache=True)
def generate_pol_noise(seed, std, t_start_rel, sample_rate, n):
    """Per-pol independent receiver noise. Called separately for V and H
    with different seeds — preserves the V/H independence fix from
    earlier, just via a different underlying RNG algorithm (see caveat
    above)."""
    out_real = np.empty(n, dtype=np.float64)
    out_imag = np.empty(n, dtype=np.float64)
    start_sample_abs = t_start_rel * sample_rate
    for i in prange(n):
        sample_idx_abs = int(start_sample_abs + i)
        g_real, g_imag = _gaussian_pair(seed, sample_idx_abs)
        out_real[i] = g_real * std
        out_imag[i] = g_imag * std
    return out_real + 1j * out_imag


# ============================================================
# WIDEBAND CHANNELIZER — produces ALL channels per call
# ============================================================


class WidebandChannelizer:
    """
    Two performance changes from an earlier version, both benchmarked on
    real hardware as targeted fixes for the confirmed channelization
    bottleneck (~70-74% of the tick once generation was numba-accelerated):

    1. No more fftshift on the (n_new, fft_len) spectra array. fftshift
       is a full data reorder (implemented via roll) over the whole
       array every call — real cost for no numerical reason, since
       "ascending frequency order" is just a labeling convention, not
       something the math needs. Internally, channels now stay in
       natural FFT bin order. Externally, NOTHING CHANGES: a fixed
       permutation (computed once at construction, applied only to the
       small channel_id integers, never to the bulk data) maps natural
       bin index back to the same external channel_id/frequency ordering
       the old fftshifted version produced — verified byte-for-byte
       against the old implementation, see correctness check.
    2. complex64 instead of complex128 internally. Same reasoning
       already applied to the tone table and noise generator: output is
       8-bit quantized regardless, so complex128's extra precision
       through 3-4 full-array passes (window multiply, FFT, phase
       correction) buys nothing while costing real memory bandwidth.
       This IS a precision reduction, not free — same category of
       change as the tone/noise ones, not the same category as the
       fftshift removal (which changes nothing numerically).
    """

    def __init__(self, fft_len: int = FFT_LEN, overlap: int = OVERLAP):
        self.fft_len = fft_len
        self.overlap = overlap
        self._history = (
            np.zeros(overlap, dtype=np.complex64)
            if overlap
            else np.zeros(0, dtype=np.complex64)
        )
        self._window = (np.hanning(fft_len) if overlap else np.ones(fft_len)).astype(
            np.float32
        )

        # Permutation from NATURAL FFT bin order -> the external channel_id
        # ordering the old fftshifted version produced. Computed once;
        # applied only to small integer arrays downstream (in
        # channel_center_frequencies and by the caller), never to the
        # bulk spectra data.
        self._natural_to_external = np.argsort(np.fft.fftshift(np.arange(fft_len)))

    def process(self, block: np.ndarray) -> np.ndarray:
        """Returns shape (n_new_samples, fft_len), columns in NATURAL FFT
        bin order (not shifted to ascending-frequency order — see class
        docstring). Caller must use channel_center_frequencies() below
        (which is in the same natural order) for any per-column frequency
        use, and the natural_to_external permutation when labeling
        channels externally."""
        extended = np.concatenate([self._history, block]) if self.overlap else block
        extended = extended.astype(np.complex64)
        if self.overlap:
            self._history = extended[-self.overlap :]

        step = self.fft_len - self.overlap
        if len(extended) < self.fft_len:
            return np.empty((0, self.fft_len), dtype=np.complex64)

        from numpy.lib.stride_tricks import sliding_window_view

        windows = sliding_window_view(extended, self.fft_len)[
            ::step
        ]  # (n_ffts, fft_len)
        windows = windows * self._window  # broadcast over rows
        # scipy.fft, not numpy.fft: numpy.fft.fft has NO multi-threading
        # support at any array size — this is the actual lever for the
        # confirmed channelization bottleneck (the fftshift-removal and
        # complex64 changes made no measurable difference on real
        # hardware; the FFT computation itself, not memory movement
        # around it, appears to be the real cost).
        spectra = scipy.fft.fft(
            windows, axis=-1, workers=FFT_WORKERS
        )  # natural bin order — no fftshift
        return spectra

    def channel_center_frequencies(
        self, base_freq_hz: float, sample_rate: float
    ) -> np.ndarray:
        """NATURAL FFT bin order — matches process()'s output column
        order. NOT ascending frequency order; use natural_to_external
        for that when labeling channels externally."""
        freqs = np.fft.fftfreq(self.fft_len, d=1.0 / sample_rate)
        return base_freq_hz + freqs


# ============================================================
# RING BUFFER — fixes a perf bug from an even earlier version, where
# np.roll(hist, ...) copied the ENTIRE history buffer (tens of thousands
# of samples) on every tick regardless of how few new samples arrived.
# Both write() and read_window() here only touch the samples actually
# involved — O(n_new) and O(length), never O(capacity).
# ============================================================


class RingBuffer:
    """Fixed-capacity circular buffer of complex samples."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf = np.zeros(capacity, dtype=np.complex128)
        self._write_pos = 0  # next index to write into
        self.total_written = 0  # monotonically increasing sample count

    def write(self, samples: np.ndarray):
        n = len(samples)
        if n > self.capacity:
            raise ValueError(
                f"write of {n} samples exceeds buffer capacity {self.capacity}"
            )
        end = self._write_pos + n
        if end <= self.capacity:
            self._buf[self._write_pos : end] = samples
        else:
            first_len = self.capacity - self._write_pos
            self._buf[self._write_pos :] = samples[:first_len]
            self._buf[: end - self.capacity] = samples[first_len:]
        self._write_pos = end % self.capacity
        self.total_written += n

    def read_window(self, length: int, end_offset: int = 0) -> Optional[np.ndarray]:
        """`length` contiguous samples ending `end_offset` samples before
        the most recently written sample (end_offset=0 -> window ends at
        the most recent sample). None if not enough history yet, or if
        end_offset is negative (would require samples not yet written —
        see caveat below)."""
        if end_offset < 0:
            # A negative offset would mean "the corrected signal needs
            # samples that haven't arrived yet" — not physically producible
            # in a causal streaming system. If you hit this in practice,
            # it likely means a sign-convention mismatch in how coarse_shift
            # is computed/applied, not a buffer sizing problem — worth
            # checking against the delay-direction assumption flagged
            # earlier in generate_next_chunk before assuming this is fine
            # to just clip.
            log.warning(
                "negative end_offset=%d requested from ring buffer — "
                "check delay sign convention",
                end_offset,
            )
            return None
        if self.total_written < length + end_offset:
            return None
        end_abs = (self._write_pos - end_offset) % self.capacity
        start_abs = (end_abs - length) % self.capacity
        if start_abs < end_abs:
            return self._buf[start_abs:end_abs].copy()
        return np.concatenate([self._buf[start_abs:], self._buf[:end_abs]])


# ============================================================
# STATION STREAMER — generation only, produces raw channelized samples
# for one pol per call (all channels). No packing here.
# ============================================================


class StationStreamer:
    def __init__(
        self,
        station: StationConfig,
        source_cfgs: list[dict],
        executor: ThreadPoolExecutor,
        obs_time_ref: float,
        noise_cfg: Optional[dict] = None,
        sample_rate: float = SAMPLE_RATE_HZ,
        max_expected_delay_s: float = 1e-3,
    ):
        self.station = station
        self.sample_rate = sample_rate
        self._executor = executor  # now only used for the channelization wave (2 tasks/tick) — see below
        self._obs_time_ref = obs_time_ref

        # --- Split source_cfgs for the numba fast path (tone/pulsed) vs
        # fallback (anything else — currently only kind='noise' as a
        # SHARED source, distinct from per-pol station noise below).
        # See the scoping note above generate_shared_signal() for why.
        tone_cfgs = [c for c in source_cfgs if c["kind"] == "tone"]
        pulse_cfgs = [c for c in source_cfgs if c["kind"] == "pulsed"]
        fallback_cfgs = [c for c in source_cfgs if c["kind"] not in ("tone", "pulsed")]

        self._tone_freqs = np.array([c["freq_hz"] for c in tone_cfgs], dtype=np.float64)
        self._tone_amps = np.array(
            [c.get("amplitude", 1.0) for c in tone_cfgs], dtype=np.float64
        )
        self._pulse_periods = np.array(
            [c["period_s"] for c in pulse_cfgs], dtype=np.float64
        )
        self._pulse_widths = np.array(
            [c["width_s"] for c in pulse_cfgs], dtype=np.float64
        )
        self._pulse_amps = np.array(
            [c.get("amplitude", 1.0) for c in pulse_cfgs], dtype=np.float64
        )
        self._tone_table = build_tone_table_unit()

        self._fallback_sources = [build_source(c) for c in fallback_cfgs]
        if fallback_cfgs:
            log.warning(
                "source_cfgs contains kind(s) %s not covered by the numba fast path "
                "(only 'tone'/'pulsed' are accelerated) — falling back to the original "
                "unaccelerated implementation for these.",
                sorted(set(c["kind"] for c in fallback_cfgs)),
            )

        # Station (receiver) noise — now generated via the numba kernel,
        # NOT the old Philox-based NoiseSource. Different algorithm, same
        # statistical properties, NOT bit-identical to old output (see
        # scoping note above). V/H independence (fixed earlier) preserved
        # via different seeds, same as before.
        self._noise_cfg = noise_cfg
        self._noise_seed_v = noise_cfg["seed"] if noise_cfg else 0
        self._noise_seed_h = (noise_cfg["seed"] + 1_000_003) if noise_cfg else 0
        self._noise_std = noise_cfg["std"] if noise_cfg else 0.0

        max_delay_samples = int(np.ceil(max_expected_delay_s * sample_rate))
        block_len_hint = int(sample_rate * BLOCK_DURATION_S)
        ring_capacity = max_delay_samples + OVERLAP + 2 * block_len_hint
        self.history_buffer = {
            "V": RingBuffer(ring_capacity),
            "H": RingBuffer(ring_capacity),
        }
        self.channelizer = {"V": WidebandChannelizer(), "H": WidebandChannelizer()}

        self._current_poly: Optional[DelayPolynomial] = None

    @property
    def channel_id_map(self) -> np.ndarray:
        """Maps a NATURAL FFT bin index to the external ascending-frequency
        channel_id — see WidebandChannelizer's docstring. ScanRunner
        (common.py) reads this uniformly across both streamer backends."""
        return self.channelizer["V"]._natural_to_external

    @property
    def num_channels(self) -> int:
        return self.channelizer["V"].fft_len

    def tick_n_samples(self) -> int:
        """Wideband INPUT samples for one tick — sized so channelization
        produces close to exactly one heap's worth of per-channel output
        samples (BLOCK_DURATION_S is defined for exactly this)."""
        return int(self.sample_rate * BLOCK_DURATION_S)

    def _refresh_delay_poly_if_needed(self, t: float):
        if self._current_poly is None or t >= self._current_poly.valid_until:
            self._current_poly = fetch_delay_model_from_cbf(self.station.station_id, t)

    def _generate_shared_and_fallback(self, t_start_rel: float, n: int) -> np.ndarray:
        """Shared (pol-independent) tone+pulse content via numba, plus any
        fallback-kind sources (numpy, unaccelerated) added on top. Called
        ONCE per tick, not once per pol — see generate_shared_signal's
        docstring for why that's a fix, not just a rename."""
        shared = generate_shared_signal(
            self._tone_table,
            self._tone_table.shape[0],
            self._tone_freqs,
            self._tone_amps,
            self._pulse_periods,
            self._pulse_widths,
            self._pulse_amps,
            self.sample_rate,
            t_start_rel,
            n,
        )
        for source in self._fallback_sources:
            shared = shared + source.generate(t_start_rel, n, self.sample_rate)
        return shared

    def _channelize_and_correct(
        self, pol: str, shifted_block: np.ndarray, tau_frac_s: float
    ) -> Optional[np.ndarray]:
        """Runs in a worker thread. Each pol has its own WidebandChannelizer
        instance (own overlap-history state), so two pols' calls never
        touch shared mutable state — safe to run concurrently. This is
        now the ONLY thing the executor is used for."""
        spectra = self.channelizer[pol].process(shifted_block)
        if spectra.shape[0] == 0:
            return None
        freqs = self.channelizer[pol].channel_center_frequencies(
            BASE_FREQ_HZ, self.sample_rate
        )
        return spectra * np.exp(-1j * 2 * np.pi * freqs * tau_frac_s)[np.newaxis, :]

    def generate_next_tick(
        self, t: float, n_input_samples: int
    ) -> dict[str, Optional[np.ndarray]]:
        """Generates BOTH pols for one tick. Generation (tone/pulse/noise)
        is direct numba calls, internally parallelized via prange — no
        ThreadPoolExecutor futures for this stage at all. The executor is
        still used, but only for the channelization wave (2 concurrent
        tasks, V and H)."""
        self._refresh_delay_poly_if_needed(t)

        # t_rel stays small (bounded by scan duration) — see earlier
        # precision-bug note; delay-poly evaluation below still uses
        # absolute t, which already handles its own reference internally.
        t_rel = t - self._obs_time_ref

        # Shared tone/pulse content: ONE call for both pols (see docstring
        # on generate_shared_signal for why this isn't per-pol).
        shared = self._generate_shared_and_fallback(t_rel, n_input_samples)

        raw = {}
        for pol, seed in (("V", self._noise_seed_v), ("H", self._noise_seed_h)):
            if self._noise_cfg is not None:
                noise = generate_pol_noise(
                    seed, self._noise_std, t_rel, self.sample_rate, n_input_samples
                )
                raw[pol] = shared + noise
            else:
                raw[pol] = shared

        # --- delay lookup + ring buffer bookkeeping: cheap, sequential is fine ---
        chan_inputs = {}
        results: dict[str, Optional[np.ndarray]] = {}
        for pol in ("V", "H"):
            tau_s = self._current_poly.eval_delay_seconds(t, pol)
            coarse_shift, tau_frac_s = split_coarse_fine(tau_s, self.sample_rate)
            ring = self.history_buffer[pol]
            ring.write(raw[pol])
            shifted_block = ring.read_window(n_input_samples, end_offset=coarse_shift)
            if shifted_block is None:
                log.warning(
                    "not enough history for coarse delay=%d samples, pol=%s",
                    coarse_shift,
                    pol,
                )
                results[pol] = None
                continue
            chan_inputs[pol] = (shifted_block, tau_frac_s)

        # --- Wave 2: channelization, one task per pol, concurrent ---
        chan_futures = {
            pol: self._executor.submit(
                self._channelize_and_correct, pol, block, tau_frac_s
            )
            for pol, (block, tau_frac_s) in chan_inputs.items()
        }
        for pol, fut in chan_futures.items():
            results[pol] = fut.result()

        return results
