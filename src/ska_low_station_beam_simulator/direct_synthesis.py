"""
Direct per-channel synthesis — the SOLE signal-generation backend for
this simulator. Bypasses wideband time-domain generation + FFT
channelization for the per-tick hot path entirely, for all three source
types this simulator supports: TONE, per-pol station (receiver) NOISE,
and PULSED (pulsar) sources.

THIS MODULE IS A CONVERGENCE of four earlier, separate pieces (see git
history for their individual development):
    - This module's own original scope (tone + live-generated noise).
    - tiled_noise_streamer.py's pre-generated noise TILE BANK, adopted
      here as the noise strategy (replacing live per-tick Box-Muller
      generation) specifically so a station pod can meet a small
      (~8 core) CPU budget at full 448-channel band -- see the tile-bank
      section below for the resource/fidelity tradeoff this involves.
    - pulsed_source_streamer.py's coherent, dispersion-aware pulsar
      generation (v3 of that module's design -- see its section below
      for why v1 and v2 were wrong).
    - wideband_streamer.py's StationStreamer, the legacy wideband+FFT
      path, is now DELETED. It was kept only as a fallback for pulsed
      sources; now that pulsed sources have a direct per-channel
      representation, nothing in this simulator needs the wideband+FFT
      path at all. WidebandChannelizer's one remaining use (a one-time,
      offline channelization step for pulsar template construction) is
      reimplemented as a small standalone function below
      (_channelize_once) rather than kept as a live dependency on a
      deleted module.

WHY DIRECT SYNTHESIS AT ALL (the original rationale, still the reason
none of the three source types touch a per-tick FFT):
    - Per-tick work for a wideband+FFT approach scales as O(NUM_CHANNELS)
      for generation, worse than O(NUM_CHANNELS) for the FFT (N log N).
    - The per-tick TIME BUDGET is fixed (BLOCK_DURATION_S =
      HEAP_LEN/CHANNEL_WIDTH_HZ), independent of channel count.
    - Scaling from 96 channels (75 MHz) to 448 channels (350 MHz, full
      SKA-Low band) is a large increase in work against an unchanged
      budget. Direct, closed-form (or precomputed-and-replayed) per-
      channel generation is what makes 448 channels viable at all on
      real target hardware -- see CLAUDE.md's Benchmarking section.

============================================================
TONE
============================================================
Spectrally sparse — after channelization it lives almost entirely in ONE
channel. Synthesized directly via a closed-form complex exponential at
the tone's frequency residual relative to that channel's center. Cost is
O(1) per tone — INDEPENDENT of total channel count. Delay is a
CONTINUOUS PHASE TERM applied directly in the exponent — no ring buffer,
no coarse/fine integer-sample split. This is EXACT for a truly
monochromatic tone (verified in __main__ to ~1e-9), not an approximation.

============================================================
NOISE — pre-generated tile bank (adopted from tiled_noise_streamer.py)
============================================================
Once the bank exists, per-tick cost is an index hash + a memcopy —
independent of channel count and, past a couple of threads, independent
of CPU budget too (benchmarked: ~8 cores clears the 448-channel budget
with room to spare; even 1-2 would). This is NOT a free upgrade over
live Box-Muller generation — it is a deliberate fidelity/resource
tradeoff, and the tradeoff is REPEATS, not noise or CPU cost:

  - By the birthday paradox, a station's OWN tile-index sequence hits
    its first repeat after roughly 1.25*sqrt(n_tiles) ticks. For any
    memory-feasible n_tiles (a few hundred to a few thousand — a tile is
    tile_n_samples * num_channels * 16 bytes per pol), that's seconds,
    not minutes: a station will replay its small fixed set of tiles many
    times over a real scan.
  - This is FINE for tests that only care about per-tick and
    cross-station statistics (delay-tracking, correlation, beamforming
    functional correctness) but WRONG for any test that checks a single
    station's long-integration noise-floor behaviour (total power should
    keep averaging down with more integration time — it won't, past the
    bank's repeat cycle). Confirm this doesn't matter for the tests this
    simulator needs to support.
  - CROSS-STATION INDEPENDENCE IS PRESERVED BY CONSTRUCTION: each
    station's tile index is drawn from that station's own (station, pol)
    noise seed — NEVER from a value shared across stations. Sharing the
    index (e.g. seeding the draw from obs_time alone) would make every
    station emit byte-identical "noise" for a given tick, silently
    breaking beamforming-SNR and cross-correlation tests that rely on
    receiver noise being uncorrelated between stations.

`n_tiles` and `tile_n_samples` ("tile width") are both configurable —
see DirectSynthesisStreamer's constructor. Bigger tiles or more of them
cost more one-time build time and memory (linear in both) and push the
first-repeat point out (roughly as sqrt(n_tiles), NOT linearly — no
memory-feasible size makes repeats rare over a full scan; it only delays
them a bit).

Also fixes a real physics bug present in the now-deleted
wideband_streamer.py: station (receiver) noise there got delay-corrected
identically to the sky signal, which is wrong — receiver noise
originates locally at each station, after any signal-path delay would
apply. Noise never enters a delay pipeline here at all (by construction,
not as a patch).

============================================================
PULSED (pulsar) sources — v3 design (from pulsed_source_streamer.py)
============================================================
A pulsar is genuinely periodic, unlike noise: replaying one precomputed
period isn't a fidelity compromise, it's ground truth. So there is no
birthday-paradox tradeoff for pulsars — periodicity here is exact, to
the precision this simulator needs.

THIS WENT THROUGH THREE DESIGNS before arriving at the current one — the
history matters because each wrong version looked reasonable until it
was actually built and numerically checked, not just reasoned about:

  v1 (WRONG): each channel sees one constant DM delay (evaluated at that
  channel's center frequency only), applied to a real-valued achromatic
  envelope; per-tick geometric delay via a first-order Taylor correction.
  v2 (fixed intra-channel smear, still broken for beamforming): a
  channel isn't one frequency, it's a ~781kHz passband — at SKA-Low
  frequencies, even DM=2 pc/cm^3 (a low, realistic value) smears the
  dispersion curve across tens of thousands of channel-widths at the
  bottom of the band, a real low-frequency-radio effect, not a bug. v2
  fixed this by averaging many shifted copies of the profile across each
  channel's own passband. But v2 was still real-valued (no carrier),
  which turns out to be a bigger problem: CBF's beamformer coherently
  combines stations by phase-rotating already-channelized complex data —
  physically valid only because a real channelizer inherently produces
  complex output with genuine carrier phase. Real-valued content has no
  phase for that rotation to act on, so v2 could not be coherently
  beamformed across stations at all (tone doesn't have this problem — it
  has a genuine residual-frequency carrier by construction).
  v3 (current): generate the wideband, undispersed pulse train as one
  real time series spanning the whole band (a shared "sky carrier" — see
  below), apply the standard coherent-dispersion transfer function
  (Lorimer & Kramer 2006, eq. 5.21) directly to its full complex FFT,
  then channelize (_channelize_once, a one-time, offline call — not a
  hot-path FFT). This fixes both v1/v2 problems at once: intra-channel
  smear falls out correctly as an emergent property of dispersing at
  full wideband FFT resolution before channelizing, and channelizing a
  real signal via FFT inherently produces genuinely complex per-channel
  content with real carrier phase — which lets the per-tick geometric-
  delay correction use tone's EXACT phase trick instead of a Taylor
  approximation.

Verified against an external, peer-reviewed reference, not just internal
self-consistency: this module's dispersion constant (4148.808) matches
NANOGrav's PsrSigSim package's DM_K (1/2.41e-4 = 4149.38) to 0.014% —
the same standard literature constant, cross-validated independently
(PsrSigSim itself wasn't taken as a runtime dependency: its own package
is heavy and partially broken for this purpose — pulls in PINT, fitsio,
emcee, nestle, matplotlib just to import; its BasebandSignal.to_RF/
to_FilterBank conversions are unimplemented stubs — so its
ISM._disperse_baseband physics was read directly and reimplemented here
instead).

WHY THE "SKY CARRIER" IS SHARED ACROSS STATIONS, NOT PER-STATION —
opposite of the rule for noise, easy to get backwards: every station in
a real array observes the literal same wavefront from the same source,
just arriving at a different time because of geometry — that's the
entire physical basis of interferometry. So the wideband pulse train's
random carrier is generated from a single fixed seed shared by every
station simulating this pulsar, never from station.station_id. Receiver
noise is the opposite: independently seeded per station, because each
station's receiver is a physically separate noise source.

WHAT PULSED SOURCES DO NOT MODEL: pulse-to-pulse jitter, scintillation,
nulling, profile evolution with frequency, or realistic flux/SNR
calibration against the receiver noise floor — the last of these
matters specifically if the goal is testing whether PSS/PST can actually
detect the injected pulsar as a candidate, not just exercising
delay-tracking.

============================================================
NUMBA
============================================================
Benchmarked directly against plain numpy alternatives before committing
to it (see CLAUDE.md's benchmarking notes) — this earns its complexity:
    - Noise-bank build (Box-Muller): numba+prange is the only way to
      actually parallelize this (numpy's Generator has no built-in
      multi-threading).
    - Tone and per-tick tile-bank/pulsar serving: numba's fused loops
      avoid full-array temporaries; the pulsar per-tick kernel
      specifically needed a phase-accumulator (NCO-style) recurrence
      rather than calling cos/sin per (sample, channel) — that mistake
      alone needed 4x the threads to clear budget, the same
      "per-element transcendental calls are expensive" lesson the noise
      kernel work already established.
Philox isn't used for the RNG (not supported in numba nopython mode) —
an independent splitmix64-style hash + Box-Muller implementation is used
instead throughout.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
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

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)

DEFAULT_N_TILES = 256

# Dispersion constant D (Lorimer & Kramer 2004/2006, eq. 5.1/5.21):
#   t_DM[s] = D * DM[pc cm^-3] / f[MHz]^2
# Cross-validated in __main__ against PsrSigSim's DM_K = 1/2.41e-4 = 4149.38
# (same standard constant, matches to within literature-typical precision).
DISPERSION_CONST_S_MHZ2_PER_DM = 4148.808

# common.py's BASE_FREQ_HZ is an explicit placeholder (0.0) pending ICD
# confirmation — dispersion delay diverges as f -> 0, so a pulsed source
# needs a real band-start frequency to be physically meaningful (enforced
# in DirectSynthesisStreamer.__init__). Default to a plausible SKA-Low
# value for pulsar-only standalone use (e.g. build_pulsar_template calls
# in tests); override via base_freq_hz for real use.
DEFAULT_PULSAR_BASE_FREQ_HZ = 50e6  # 50 MHz, illustrative SKA-Low low-band edge

# The pulsar's intrinsic "sky carrier" seed -- shared by EVERY station
# simulating this pulsar (see module docstring). NEVER derive this from
# station.station_id.
DEFAULT_SKY_SEED = 0x5AB1E5EED


# ============================================================
# CORE KERNELS: splitmix64 hash + Box-Muller, delay-polynomial eval, tone
# ============================================================


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
def _gaussian_pair(seed, index):
    u1 = _uniform_from_hash(_splitmix64_hash(seed, 2 * index))
    u2 = _uniform_from_hash(_splitmix64_hash(seed, 2 * index + 1))
    u1 = max(u1, 1e-300)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    return r * np.cos(theta), r * np.sin(theta)


@njit(cache=True)
def eval_delay_poly_ns(coeffs, t_rel):
    """t_rel MUST already be relative to the polynomial's own
    start_validity_sec (small magnitude) — same precision-safety
    requirement established for common.DelayPolynomial."""
    tau_ns = 0.0
    power = 1.0
    for c in range(coeffs.shape[0]):
        tau_ns += coeffs[c] * power
        power *= t_rel
    return tau_ns


@njit(cache=True)
def synth_tone_channel(
    freq_hz,
    amplitude,
    base_freq_hz,
    channel_width_hz,
    delay_coeffs,
    poly_t_rel_start,
    ypol_offset_ns,
    is_h_pol,
    t_local_rel_start,
    sample_rate_per_channel,
    n_samples,
):
    """
    Returns (channel_index, samples). Delay applied as continuous phase
    modulation — no ring buffer, no coarse/fine split. poly_t_rel_start
    and t_local_rel_start are both small-magnitude relative times (see
    module docstring's note on why raw epoch time breaks this).
    """
    channel_idx = int(round((freq_hz - base_freq_hz) / channel_width_hz))
    channel_center = base_freq_hz + channel_idx * channel_width_hz
    residual_freq = freq_hz - channel_center

    samples = np.empty(n_samples, dtype=np.complex128)
    for i in prange(n_samples):
        t_local = t_local_rel_start + i / sample_rate_per_channel
        t_poly = poly_t_rel_start + i / sample_rate_per_channel
        tau_ns = eval_delay_poly_ns(delay_coeffs, t_poly)
        if is_h_pol:
            tau_ns += ypol_offset_ns
        tau_s = tau_ns * 1e-9
        phase = 2.0 * np.pi * residual_freq * t_local - 2.0 * np.pi * freq_hz * tau_s
        samples[i] = amplitude * np.exp(1j * phase)
    return channel_idx, samples


# ============================================================
# NOISE — Box-Muller kernels, used to fill the tile bank (once, at
# construction) rather than per tick. See module docstring.
# ============================================================


@njit(parallel=True, cache=True)
def synth_noise_all_channels(seed, std, sample_index_start, num_channels, n_samples):
    """
    Independent complex Gaussian directly at (n_samples, num_channels)
    resolution — statistically exact equivalent of "wideband noise, then
    FFT channelized" (DFT of i.i.d. Gaussian is i.i.d. Gaussian). NO
    delay applied — physically correct for receiver noise.
    """
    out_real = np.empty((n_samples, num_channels), dtype=np.float64)
    out_imag = np.empty((n_samples, num_channels), dtype=np.float64)
    for i in prange(n_samples):
        for ch in range(num_channels):
            flat_index = (sample_index_start + i) * num_channels + ch
            g_real, g_imag = _gaussian_pair(seed, flat_index)
            out_real[i, ch] = g_real * std
            out_imag[i, ch] = g_imag * std
    return out_real + 1j * out_imag


@njit(parallel=True, cache=True)
def synth_noise_all_channels_into(out, seed, std, sample_index_start, num_channels, n_samples):
    """Same statistics as synth_noise_all_channels, but WRITES directly
    into a caller-provided (n_samples, num_channels) complex128 buffer
    instead of allocating fresh arrays — used to fill each tile bank
    entry once, at construction. Writes (not +=) every cell."""
    for i in prange(n_samples):
        for ch in range(num_channels):
            flat_index = (sample_index_start + i) * num_channels + ch
            g_real, g_imag = _gaussian_pair(seed, flat_index)
            out[i, ch] = (g_real * std) + 1j * (g_imag * std)


def bank_memory_bytes(n_tiles: int, tile_n_samples: int, num_channels: int, n_pols: int = 2) -> int:
    """Total resident memory for the noise bank across both pols."""
    return n_tiles * tile_n_samples * num_channels * 16 * n_pols  # complex128 = 16 bytes


@njit(parallel=True, cache=True)
def _copy_tile_into(out, bank, tile_idx):
    """out[:] = bank[tile_idx], parallelized over rows -- lets the
    benchmark test whether spreading the copy across threads matters at
    all once generation itself is no longer in the per-tick path."""
    n_samples = out.shape[0]
    num_channels = out.shape[1]
    for i in prange(n_samples):
        for ch in range(num_channels):
            out[i, ch] = bank[tile_idx, i, ch]


# ============================================================
# PULSAR — wideband generate + coherent dispersion + one-time
# channelization. See module docstring's v1/v2/v3 history.
# ============================================================


def dispersion_delay_s(freq_hz: float, dm_pc_cm3: float) -> float:
    freq_mhz = freq_hz / 1e6
    return DISPERSION_CONST_S_MHZ2_PER_DM * dm_pc_cm3 / (freq_mhz**2)


@njit(cache=True)
def _real_gaussian(seed, index):
    g_real, _ = _gaussian_pair(seed, index)
    return g_real


@njit(cache=True)
def _gaussian_envelope(t, peak, period_s, sigma, amplitude):
    d = t - peak
    d = d - period_s * np.floor(d / period_s + 0.5)  # wrap into [-period/2, period/2)
    return amplitude * np.exp(-0.5 * (d / sigma) ** 2)


@njit(cache=True)
def generate_wideband_pulse_train(seed, period_s, width_s, amplitude, wideband_rate, n_wide):
    """The shared 'sky carrier': one period of a real, wideband (spans
    the WHOLE band as a single time series), UNDISPERSED pulse train.
    Physically: incoherent broadband radio emission (noise-like),
    power-modulated by the pulsar's rotation -- amplitude = envelope(t)
    * a real Gaussian draw, same model PsrSigSim's _make_amp_pulses
    uses. Deterministic/seekable from (seed, sample index) alone, same
    property every other kernel in this codebase relies on."""
    sigma = width_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # width_s = FWHM
    peak = period_s / 2.0
    v = np.empty(n_wide, dtype=np.float64)
    for i in range(n_wide):
        t = i / wideband_rate
        env = _gaussian_envelope(t, peak, period_s, sigma, amplitude)
        v[i] = env * _real_gaussian(seed, i)
    return v


def _channelize_once(v: np.ndarray, num_channels: int) -> np.ndarray:
    """One-shot FFT channelization of a complex 1D array of length
    n = k*num_channels into (k, num_channels), EXTERNAL ascending-
    channel-id order (channel c's center frequency is
    base_freq_hz + c*channel_width_hz for whatever base_freq_hz the
    caller used to build v's frequency axis).

    Equivalent to the now-deleted wideband_streamer.WidebandChannelizer
    at overlap=0 for a single, already-complete block -- ported as a
    minimal standalone function since this is the one remaining use of
    that class, and it was only ever needed for this one-time, offline
    pulsar-template construction step, never the per-tick hot path. With
    overlap=0 and step==fft_len, WidebandChannelizer's sliding-window
    approach is exactly equivalent to a reshape (no windows actually
    slide past each other), which is what this does directly.
    """
    n_out = v.shape[0] // num_channels
    v = v[: n_out * num_channels]
    windows = v.reshape(n_out, num_channels)
    spectra = np.fft.fft(windows, axis=-1)  # natural FFT bin order
    natural_to_external = np.argsort(np.fft.fftshift(np.arange(num_channels)))
    external = np.empty_like(spectra)
    external[:, natural_to_external] = spectra
    return external


def build_pulsar_template(
    num_channels: int,
    channel_width_hz: float,
    base_freq_hz: float,
    channel_output_rate: float,
    period_s: float,
    width_s: float,
    amplitude: float,
    dm_pc_cm3: float,
    sky_seed: int = DEFAULT_SKY_SEED,
):
    """ONE-TIME, offline construction (plain numpy -- no per-tick budget
    applies here, so a real FFT is fine, unlike the hot path). Generates
    the shared wideband pulse train, applies the coherent dispersion
    transfer function directly to its full complex FFT (capturing
    intra-channel smear as an emergent property of the full wideband
    frequency resolution), then channelizes via _channelize_once to
    produce genuinely complex per-channel content with real carrier
    phase.

    Returns (template, n_period_samples): template is
    (num_channels, n_period_samples) complex128, in this project's
    EXTERNAL ascending-channel-id order -- verified against a known-tone
    injection check in __main__, not assumed from reading the FFT-bin
    permutation logic.
    """
    wideband_rate = num_channels * channel_width_hz
    n_period_samples = int(round(period_s * channel_output_rate))
    n_wide = n_period_samples * num_channels  # exact multiple of num_channels,
    # required so _channelize_once's reshape doesn't drop a partial block.

    v = generate_wideband_pulse_train(sky_seed, period_s, width_s, amplitude, wideband_rate, n_wide)

    # Full complex FFT, NOT rfft -- this project's own convention (see
    # _channelize_once / the deleted WidebandChannelizer.channel_center_frequencies)
    # treats negative fftfreq bins as meaningful, independent channels —
    # using rfft here originally only covered half the intended band and
    # shifted every channel's frequency. Caught by injecting a KNOWN tone
    # at a known external channel and checking where it actually landed,
    # not by reasoning about the convention. v_dispersed is genuinely
    # complex after this (dispersion breaks the real signal's Hermitian
    # symmetry) -- expected, not a bug, and what gives the template real
    # carrier phase.
    V = np.fft.fft(v)
    u = np.fft.fftfreq(n_wide, d=1.0 / wideband_rate)  # natural bin order, [-wideband_rate/2, wideband_rate/2)
    band_center_hz = base_freq_hz + wideband_rate / 2.0
    f_offset_mhz = u / 1e6
    f0_mhz = band_center_hz / 1e6
    # Lorimer & Kramer 2006, eq. 5.21 -- coherent dispersion transfer
    # function, cross-validated against PsrSigSim's ISM._disperse_baseband.
    H = np.exp(
        1j * 2 * np.pi * DISPERSION_CONST_S_MHZ2_PER_DM
        / ((f_offset_mhz + f0_mhz) * f0_mhz**2) * dm_pc_cm3 * f_offset_mhz**2
    )
    v_dispersed = np.fft.ifft(V * H)

    channelized = _channelize_once(v_dispersed, num_channels)  # (n_period_samples, num_channels), external order
    template = np.ascontiguousarray(channelized.T)  # (num_channels, n_period_samples)
    return template, n_period_samples


@njit(parallel=True, cache=True)
def add_pulsar_tick(
    out, template, start_idx, n_period_samples,
    delay_coeffs, poly_t_rel_start, ypol_offset_ns, is_h_pol,
    base_freq_hz, channel_width_hz, channel_output_rate, n_samples, num_channels,
):
    """Adds this tick's contribution into `out` (n_samples, num_channels
    complex128), reading the precomputed template circularly and
    applying the EXACT per-channel geometric-delay phase correction --
    valid because the template is genuinely complex/narrowband per
    channel (see module docstring), the same reasoning that makes
    synth_tone_channel's delay-as-phase exact. Adds (not writes) so this
    composes with noise/tone in generate_next_tick's fixed order.

    Uses a phase-accumulator (NCO-style) recurrence rather than calling
    cos/sin per (sample, channel) -- phase(c) = phase(0) - c*dphase is
    linear in c at fixed sample, so each channel's rotation is the
    previous one times a single fixed per-sample step, computed via one
    complex multiply instead of two fresh transcendental calls. An
    earlier version called cos/sin per channel directly and needed 4x
    the threads to clear budget for exactly the reason this codebase's
    noise-kernel work already established: per-element transcendental
    calls are the expensive part, not the arithmetic around them."""
    for i in prange(n_samples):
        idx = (start_idx + i) % n_period_samples
        t_poly = poly_t_rel_start + i / channel_output_rate
        tau_ns = eval_delay_poly_ns(delay_coeffs, t_poly)
        if is_h_pol:
            tau_ns += ypol_offset_ns
        tau_s = tau_ns * 1e-9

        phase0 = -2.0 * np.pi * base_freq_hz * tau_s
        dphase = 2.0 * np.pi * channel_width_hz * tau_s
        rot = np.cos(phase0) + 1j * np.sin(phase0)
        step = np.cos(dphase) - 1j * np.sin(dphase)
        for c in range(num_channels):
            out[i, c] += template[c, idx] * rot
            rot = rot * step


# ============================================================
# DIRECT SYNTHESIS STREAMER — the one production streamer, all three
# source types.
#
# OUTPUT CONTRACT: generate_next_tick returns dict[pol] -> (n_samples,
# num_channels) complex array, plugging into common.HeapAccumulator/
# SpsPacketizer unchanged. Columns are already in EXTERNAL
# ascending-frequency channel_id order, so channel_id_map is identity.
# ============================================================


class DirectSynthesisStreamer:
    def __init__(
        self,
        station: StationConfig,
        source_cfgs: list[dict],
        obs_time_ref: float,
        noise_cfg: Optional[dict] = None,
        num_channels: int = NUM_CHANNELS,
        base_freq_hz: float = BASE_FREQ_HZ,
        channel_width_hz: float = CHANNEL_WIDTH_HZ,
        n_tiles: int = DEFAULT_N_TILES,
        tile_n_samples: Optional[int] = None,
        parallel_copy: bool = False,
    ):
        for cfg in source_cfgs:
            if cfg["kind"] not in ("tone", "pulsed"):
                raise ValueError(
                    f"DirectSynthesisStreamer only supports kind in "
                    f"('tone', 'pulsed') in source_cfgs; got kind={cfg['kind']!r}."
                )

        self._pulsed_cfgs = [c for c in source_cfgs if c["kind"] == "pulsed"]
        if self._pulsed_cfgs and base_freq_hz <= 0:
            raise ValueError(
                f"pulsed source configured but base_freq_hz={base_freq_hz} -- "
                f"dispersion physics diverges as frequency -> 0 (see "
                f"dispersion_delay_s). Pass a real, positive base_freq_hz "
                f"explicitly (e.g. DEFAULT_PULSAR_BASE_FREQ_HZ) — common.py's "
                f"own BASE_FREQ_HZ is an unverified ICD placeholder (0.0) and "
                f"is not usable for pulsed sources as-is."
            )

        self.station = station
        self.num_channels = num_channels
        self.base_freq_hz = base_freq_hz
        self.channel_width_hz = channel_width_hz
        # Critically sampled per-channel output rate.
        self.channel_output_rate = channel_width_hz

        self._obs_time_ref = obs_time_ref
        self._tone_cfgs = [c for c in source_cfgs if c["kind"] == "tone"]

        self._noise_cfg = noise_cfg
        self._noise_seed_v = noise_cfg["seed"] if noise_cfg else 0
        self._noise_seed_h = (noise_cfg["seed"] + 1_000_003) if noise_cfg else 0
        self._noise_std = noise_cfg["std"] if noise_cfg else 0.0

        self._current_poly: Optional[DelayPolynomial] = None
        self._delay_coeffs: Optional[np.ndarray] = None

        # --- noise tile bank (see module docstring for the tradeoff) ---
        self.n_tiles = n_tiles
        self.tile_n_samples = tile_n_samples or self.tick_n_samples()
        self.parallel_copy = parallel_copy
        self._banks: dict[str, np.ndarray] = {}
        if noise_cfg is not None:
            for pol, seed in (("V", self._noise_seed_v), ("H", self._noise_seed_h)):
                bank = np.empty(
                    (self.n_tiles, self.tile_n_samples, self.num_channels), dtype=np.complex128
                )
                for tile_idx in range(self.n_tiles):
                    synth_noise_all_channels_into(
                        bank[tile_idx],
                        seed,
                        self._noise_std,
                        tile_idx * self.tile_n_samples,  # each tile is independent content
                        self.num_channels,
                        self.tile_n_samples,
                    )
                self._banks[pol] = bank

        # --- pulsar templates: one (template, period_s, n_period_samples)
        # tuple per pulsed source cfg, built ONCE here. sky_seed is shared
        # across all stations for the same pulsar cfg by default (see
        # module docstring) -- override only for a deliberately DIFFERENT
        # (uncorrelated) pulsar, never to "vary" the same pulsar per
        # station. ---
        self._pulsars = []
        for cfg in self._pulsed_cfgs:
            template, n_period_samples = build_pulsar_template(
                self.num_channels,
                self.channel_width_hz,
                self.base_freq_hz,
                self.channel_output_rate,
                cfg["period_s"],
                cfg["width_s"],
                cfg.get("amplitude", 1.0),
                cfg["dm_pc_cm3"],
                cfg.get("sky_seed", DEFAULT_SKY_SEED),
            )
            self._pulsars.append((template, cfg["period_s"], n_period_samples))

        # Reused across ticks (per pol) so generate_next_tick doesn't
        # allocate a fresh (n_samples, num_channels) complex128 array
        # every tick. Safe to mutate in place: ScanRunner._run calls
        # generate_next_tick synchronously and HeapAccumulator.add copies
        # the data out (np.concatenate) before the next tick can run --
        # nothing downstream ever holds a reference across ticks.
        self._out_bufs: dict[str, np.ndarray] = {}

    def _get_output_buffer(self, pol: str, n_samples: int) -> np.ndarray:
        buf = self._out_bufs.get(pol)
        if buf is None or buf.shape != (n_samples, self.num_channels):
            buf = np.empty((n_samples, self.num_channels), dtype=np.complex128)
            self._out_bufs[pol] = buf
        return buf

    @property
    def channel_id_map(self) -> np.ndarray:
        return np.arange(self.num_channels)

    def tick_n_samples(self) -> int:
        """Per-channel output samples for one tick — HEAP_LEN by
        construction (BLOCK_DURATION_S is defined for exactly this)."""
        return int(round(self.channel_output_rate * BLOCK_DURATION_S))

    def bank_memory_bytes(self) -> int:
        if not self._banks:
            return 0
        return bank_memory_bytes(self.n_tiles, self.tile_n_samples, self.num_channels, len(self._banks))

    def _refresh_delay_poly_if_needed(self, t: float):
        if self._current_poly is None or t >= self._current_poly.valid_until:
            self._current_poly = fetch_delay_model_from_cbf(self.station.station_id, t)
            # Cache as a float64 array once per poly refresh, not once per
            # tick — the poly only changes when its validity window expires.
            self._delay_coeffs = np.asarray(
                self._current_poly.xypol_coeffs_ns, dtype=np.float64
            )

    def generate_next_tick(self, t: float, n_samples: int) -> dict[str, np.ndarray]:
        """n_samples is PER-CHANNEL time samples for this tick (at
        channel_output_rate). See common.ScanRunner, which sizes this via
        tick_n_samples() so it lines up with HeapAccumulator's HEAP_LEN
        framing exactly."""
        self._refresh_delay_poly_if_needed(t)
        poly = self._current_poly

        # Both t_local_rel_start (local clock for noise/tone/pulsar) and
        # poly_t_rel_start (the delay poly's own validity-relative clock)
        # must stay small-magnitude — same precision requirement as
        # everywhere else in this codebase (see common.DelayPolynomial's
        # docstring for why raw epoch-scale time breaks this).
        t_local_rel_start = t - self._obs_time_ref
        poly_t_rel_start = t - poly.start_validity_sec

        results: dict[str, np.ndarray] = {}
        for pol, is_h_pol, noise_seed in (
            ("V", False, self._noise_seed_v),
            ("H", True, self._noise_seed_h),
        ):
            out = self._get_output_buffer(pol, n_samples)

            # Noise first, written (not accumulated) so it covers every
            # cell -- that's what lets pulsar/tone below skip zeroing the
            # buffer.
            if self._noise_cfg is not None:
                bank = self._banks[pol]
                tick_index = int(round(t_local_rel_start * self.channel_output_rate)) // max(n_samples, 1)
                # Per-(station, pol) independent draw -- NEVER shared
                # across stations. See module docstring.
                tile_idx = int(_splitmix64_hash(noise_seed, tick_index) % self.n_tiles)
                if self.parallel_copy:
                    _copy_tile_into(out, bank, tile_idx)
                else:
                    out[:] = bank[tile_idx]
            else:
                out.fill(0)

            for template, period_s, n_period_samples in self._pulsars:
                phase_in_period = t_local_rel_start % period_s
                start_idx = int(round(phase_in_period * self.channel_output_rate)) % n_period_samples
                add_pulsar_tick(
                    out, template, start_idx, n_period_samples,
                    self._delay_coeffs, poly_t_rel_start, poly.ypol_offset_ns, is_h_pol,
                    self.base_freq_hz, self.channel_width_hz, self.channel_output_rate,
                    n_samples, self.num_channels,
                )

            for cfg in self._tone_cfgs:
                ch_idx, samples = synth_tone_channel(
                    cfg["freq_hz"],
                    cfg.get("amplitude", 1.0),
                    self.base_freq_hz,
                    self.channel_width_hz,
                    self._delay_coeffs,
                    poly_t_rel_start,
                    poly.ypol_offset_ns,
                    is_h_pol,
                    t_local_rel_start,
                    self.channel_output_rate,
                    n_samples,
                )
                if not (0 <= ch_idx < self.num_channels):
                    log.warning(
                        "tone freq_hz=%s maps to channel_idx=%d, outside the "
                        "configured [0, %d) channel range — skipping",
                        cfg["freq_hz"],
                        ch_idx,
                        self.num_channels,
                    )
                    continue
                out[:, ch_idx] += samples

            results[pol] = out

        return results


if __name__ == "__main__":
    # ============================================================
    # CORRECTNESS CHECKS — covers tone, noise (kernel + tile bank), and
    # pulsar (dispersion + coherence), merged from the three modules this
    # one converges.
    # ============================================================

    def _fake_fetch(station_id, at_time):
        return DelayPolynomial(
            station_id=station_id,
            start_validity_sec=at_time,
            validity_period_sec=600.0,
            xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0],
            ypol_offset_ns=2.0,
        )

    globals()["fetch_delay_model_from_cbf"] = _fake_fetch

    SAMPLE_RATE_PER_CHANNEL = CHANNEL_WIDTH_HZ  # critically sampled

    # ---------------- TONE ----------------
    test_freq = 42 * CHANNEL_WIDTH_HZ + 150_000.0  # off-center within channel 42
    zero_coeffs = np.array([0.0], dtype=np.float64)
    ch_idx, samples = synth_tone_channel(
        test_freq, 1.0, BASE_FREQ_HZ, CHANNEL_WIDTH_HZ, zero_coeffs,
        0.0, 0.0, False, 0.0, SAMPLE_RATE_PER_CHANNEL, 2048,
    )
    assert ch_idx == 42, f"expected channel 42, got {ch_idx}"
    print(f"tone channel placement: OK (channel {ch_idx})")

    residual = test_freq - (BASE_FREQ_HZ + 42 * CHANNEL_WIDTH_HZ)
    t = np.arange(2048) / SAMPLE_RATE_PER_CHANNEL
    expected = np.exp(1j * 2 * np.pi * residual * t)
    err = np.max(np.abs(samples - expected))
    print(f"tone residual-frequency phase error (zero delay): {err:.3e} (expect ~0)")
    assert err < 1e-9
    print("tone zero-delay accuracy: OK")

    known_tau_ns = 750.0
    coeffs = np.array([known_tau_ns], dtype=np.float64)
    _, samples_delayed = synth_tone_channel(
        test_freq, 1.0, BASE_FREQ_HZ, CHANNEL_WIDTH_HZ, coeffs,
        0.0, 0.0, False, 0.0, SAMPLE_RATE_PER_CHANNEL, 2048,
    )
    expected_delayed = expected * np.exp(-1j * 2 * np.pi * test_freq * known_tau_ns * 1e-9)
    err2 = np.max(np.abs(samples_delayed - expected_delayed))
    print(f"tone known-delay phase error: {err2:.3e} (expect ~0)")
    assert err2 < 1e-9
    print("tone delay-as-phase accuracy: OK (no ring buffer needed, confirmed)")

    # ---------------- NOISE KERNEL ----------------
    noise = synth_noise_all_channels(seed=7, std=1.0, sample_index_start=0, num_channels=96, n_samples=200_000)
    print(f"noise mean: {np.mean(noise):.4f} (expect ~0)")
    print(f"noise std (real, channel 0): {np.std(noise[:, 0].real):.4f} (expect ~1.0)")
    assert abs(np.mean(noise)) < 0.01
    assert abs(np.std(noise[:, 0].real) - 1.0) < 0.01
    corr = np.corrcoef(noise[:, 0].real, noise[:, 1].real)[0, 1]
    print(f"cross-channel correlation (ch0 vs ch1 real): {corr:.4f} (expect ~0)")
    assert abs(corr) < 0.02
    print("noise statistics + independence: OK")

    n1 = synth_noise_all_channels(7, 1.0, 1000, 96, 500)
    n2 = synth_noise_all_channels(7, 1.0, 1000, 96, 500)
    assert np.array_equal(n1, n2)
    print("noise determinism/seekability: OK")

    into_buf = np.full((500, 96), 999.0 + 999.0j, dtype=np.complex128)
    synth_noise_all_channels_into(into_buf, 7, 1.0, 1000, 96, 500)
    assert np.array_equal(n1, into_buf)
    print("noise in-place kernel matches allocating kernel: OK")

    # ---------------- NOISE TILE BANK (via the full streamer) ----------------
    station = StationConfig(station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1)
    tile_streamer = DirectSynthesisStreamer(
        station=station, source_cfgs=[], noise_cfg={"std": 1.0, "seed": 7},
        obs_time_ref=1_800_000_000.0, num_channels=96, n_tiles=8,
    )
    n_samples_tile = tile_streamer.tick_n_samples()
    tick_dt_tile = n_samples_tile / tile_streamer.channel_output_rate
    obs_time = 1_800_000_000.0

    r1 = tile_streamer.generate_next_tick(obs_time, n_samples_tile)["V"].copy()
    r2 = tile_streamer.generate_next_tick(obs_time, n_samples_tile)["V"].copy()
    assert np.array_equal(r1, r2), "same t must give identical content"
    print("tile-bank determinism/seekability: OK")

    seen = {}
    first_repeat_at = None
    for i in range(200):
        out = tile_streamer.generate_next_tick(obs_time + i * tick_dt_tile, n_samples_tile)["V"]
        key = out[0, 0]
        if key in seen and first_repeat_at is None:
            first_repeat_at = i
        seen[key] = i
    print(f"tile-bank first exact repeat (n_tiles=8): tick {first_repeat_at} (expect early, ~dozens)")

    tile_streamer_b = DirectSynthesisStreamer(
        station=StationConfig(station_id=2, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1),
        source_cfgs=[], noise_cfg={"std": 1.0, "seed": 99},
        obs_time_ref=1_800_000_000.0, num_channels=96, n_tiles=8,
    )
    out_a = tile_streamer.generate_next_tick(obs_time, n_samples_tile)["V"]
    out_b = tile_streamer_b.generate_next_tick(obs_time, n_samples_tile)["V"]
    assert not np.array_equal(out_a, out_b), "different stations must not emit identical noise"
    print("tile-bank cross-station independence (different seeds -> different content): OK")

    # ---------------- PULSAR: channel-mapping ground truth ----------------
    NUM_CHANNELS_TEST = 32
    CHW = CHANNEL_WIDTH_HZ
    BASE_F = DEFAULT_PULSAR_BASE_FREQ_HZ
    wideband_rate = NUM_CHANNELS_TEST * CHW
    n_wide_test = 256 * NUM_CHANNELS_TEST
    test_channel = 7
    band_center_hz_test = BASE_F + wideband_rate / 2.0
    test_freq_p = BASE_F + test_channel * CHW + 0.15 * CHW  # off-center within the channel
    f_offset_test = test_freq_p - band_center_hz_test
    t_wide = np.arange(n_wide_test) / wideband_rate
    v_tone = np.exp(1j * 2 * np.pi * f_offset_test * t_wide)
    channelized = _channelize_once(v_tone, NUM_CHANNELS_TEST)
    power_per_channel = np.mean(np.abs(channelized) ** 2, axis=0)
    detected_channel = int(np.argmax(power_per_channel))
    print(
        f"pulsar channel-mapping: tone injected at external channel {test_channel} "
        f"(freq={test_freq_p/1e6:.4f} MHz) -> detected at external channel {detected_channel}: "
        f"{'OK' if detected_channel == test_channel else 'MISMATCH -- relabeling logic is wrong'}"
    )
    assert detected_channel == test_channel

    # ---------------- PULSAR: dispersion constant cross-check ----------------
    psrsigsim_dm_k = 1.0 / 2.41e-4
    rel_diff = abs(DISPERSION_CONST_S_MHZ2_PER_DM - psrsigsim_dm_k) / psrsigsim_dm_k
    print(
        f"dispersion constant: this module={DISPERSION_CONST_S_MHZ2_PER_DM}  "
        f"PsrSigSim DM_K={psrsigsim_dm_k:.3f}  relative diff={rel_diff*100:.3f}% "
        f"(expected: small, standard literature-precision variation)"
    )

    # ---------------- PULSAR: full pipeline + coherence ----------------
    NUM_CHANNELS_TEST2 = 448
    DM_TEST = 2.0
    PERIOD_S = 0.1
    WIDTH_S = 0.005

    pulsar_streamer = DirectSynthesisStreamer(
        station=station,
        source_cfgs=[{"kind": "pulsed", "period_s": PERIOD_S, "width_s": WIDTH_S, "amplitude": 1.0, "dm_pc_cm3": DM_TEST}],
        obs_time_ref=1_800_000_000.0,
        num_channels=NUM_CHANNELS_TEST2,
        base_freq_hz=DEFAULT_PULSAR_BASE_FREQ_HZ,
    )
    n_samples_p = pulsar_streamer.tick_n_samples()

    rp1 = pulsar_streamer.generate_next_tick(obs_time, n_samples_p)["V"].copy()
    rp2 = pulsar_streamer.generate_next_tick(obs_time, n_samples_p)["V"].copy()
    assert np.array_equal(rp1, rp2), "same t must give identical content"
    print("pulsar determinism/seekability: OK")
    print(f"pulsar content is complex (not real-only): max |imag part| = {np.max(np.abs(rp1.imag)):.4f}")

    def _fake_fetch_b(station_id, at_time):
        return DelayPolynomial(
            station_id=station_id, start_validity_sec=at_time, validity_period_sec=600.0,
            xypol_coeffs_ns=[300.0, 0.002, 0.0, 0.0, 0.0, 0.0], ypol_offset_ns=1.0,
        )

    station_b = StationConfig(station_id=2, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1)
    pulsar_streamer_b = DirectSynthesisStreamer(
        station=station_b,
        source_cfgs=[{"kind": "pulsed", "period_s": PERIOD_S, "width_s": WIDTH_S, "amplitude": 1.0, "dm_pc_cm3": DM_TEST}],
        obs_time_ref=1_800_000_000.0,
        num_channels=NUM_CHANNELS_TEST2,
        base_freq_hz=DEFAULT_PULSAR_BASE_FREQ_HZ,
    )
    globals()["fetch_delay_model_from_cbf"] = _fake_fetch_b
    pulsar_streamer_b._refresh_delay_poly_if_needed(obs_time)
    globals()["fetch_delay_model_from_cbf"] = _fake_fetch
    out_b = pulsar_streamer_b.generate_next_tick(obs_time, n_samples_p)["V"]
    out_a = pulsar_streamer.generate_next_tick(obs_time, n_samples_p)["V"]

    test_ch = 200
    tau_a_ns = eval_delay_poly_ns(pulsar_streamer._delay_coeffs, obs_time - pulsar_streamer._current_poly.start_validity_sec)
    tau_b_ns = eval_delay_poly_ns(pulsar_streamer_b._delay_coeffs, obs_time - pulsar_streamer_b._current_poly.start_validity_sec)
    f_c = pulsar_streamer.base_freq_hz + test_ch * pulsar_streamer.channel_width_hz
    correction_a = np.exp(1j * 2 * np.pi * f_c * tau_a_ns * 1e-9)
    correction_b = np.exp(1j * 2 * np.pi * f_c * tau_b_ns * 1e-9)
    aligned_a = out_a[:, test_ch] * correction_a
    aligned_b = out_b[:, test_ch] * correction_b
    coh = np.abs(np.vdot(aligned_a, aligned_b)) / np.sqrt(np.vdot(aligned_a, aligned_a).real * np.vdot(aligned_b, aligned_b).real)
    print(
        f"pulsar cross-station coherence after delay-compensating phase rotation: "
        f"normalized correlation={coh:.4f} (expect close to 1.0)"
    )
    assert coh > 0.999

    print("\nAll correctness checks passed.")
