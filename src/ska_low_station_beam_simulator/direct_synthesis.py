"""
Direct per-channel synthesis — the SOLE signal-generation backend for
this simulator. Bypasses wideband time-domain generation + FFT
channelization for the per-tick hot path entirely, for all three source
types this simulator supports: TONE, per-pol station (receiver) NOISE,
and PULSED (pulsar) sources. This module's development history
(prototypes it converged from, deleted legacy paths, design iterations
that were tried and rejected) lives in CLAUDE.md, not here — this
docstring describes only the design as it stands.

WHY DIRECT SYNTHESIS AT ALL — the reason none of the three source types
touch a per-tick FFT:

    - Per-tick work for a wideband+FFT approach scales as
      O(NUM_CHANNELS) for generation, worse than O(NUM_CHANNELS) for the
      FFT (N log N).
    - The per-tick TIME BUDGET is fixed (``BLOCK_DURATION_S`` =
      ``HEAP_LEN``/``CHANNEL_OUTPUT_RATE_HZ``, the real OVERSAMPLED
      per-channel sample rate -- see ``common.py``), independent of
      channel count.
    - Scaling from 96 channels (75 MHz) to 384 channels (300 MHz, the
      real ICD maximum -- 8 to 384 in steps of 8, see CLAUDE.md's SPS-CBF
      ICD section) is a large increase in work against an unchanged
      budget. Direct, closed-form (or precomputed-and-replayed)
      per-channel generation is what makes 384 channels viable at all on
      real target hardware -- see CLAUDE.md's Benchmarking section.

TONE
====

Spectrally sparse — after channelization it lives almost entirely in ONE
channel. Synthesized directly via a closed-form complex exponential at
the tone's frequency residual relative to that channel's center. Cost is
O(1) per tone — INDEPENDENT of total channel count. Delay is a
CONTINUOUS PHASE TERM applied directly in the exponent — no ring buffer,
no coarse/fine integer-sample split. This is EXACT for a truly
monochromatic tone (verified in ``tests/test_direct_synthesis.py`` to
~1e-9), not an approximation.

OVERSAMPLING AND PER-PACKET PHASE
==================================

Reasoned through, NOT independently verified against real SPS hardware
-- flagged honestly as such. The ICD states the SPS filterbank
OVERSAMPLES by 32/27 (see ``common.CHANNEL_OUTPUT_RATE_HZ`` -- this is
now correctly reflected in ``channel_output_rate`` below, and is a real,
numerically significant fix: it shrinks the per-tick budget from
2.621ms to 2.212ms, ~15.6% tighter). The same ICD passage also says the
oversampled filterbank output "shall be derotated" and that "the first
sample of every SPS SPEAD packet has zero phase". Worked through here
rather than modeled blindly, since a literal per-heap phase RESET would
be a much bigger, more invasive change than the sample-rate fix if it
turned out to be necessary:

  - An oversampled polyphase filterbank's RAW per-channel output (before
    correction) carries a spurious, purely mechanical phase ramp from
    sample to sample -- an artifact of the analysis window advancing by
    a NON-integer number of FFT lengths per output sample (that's what
    "oversampled" means structurally), present even for a pure DC/
    channel-centre input. "Derotated" describes REMOVING that artifact
    so the channel's output is a clean, physically meaningful baseband
    signal at the new (finer) sample rate.
  - "Zero phase at the first sample of every packet" reads as the
    NORMALIZATION CONVENTION that pins down what "derotated" means in
    absolute terms: for a hypothetical channel-centre-frequency (zero
    residual) input, phase is defined to read exactly zero at each
    packet's first sample.
  - This module never generates the raw, undecorated PFB artifact at
    all -- ``synth_tone_channel``/``add_pulsar_tick`` synthesize the
    ALREADY-CLEAN baseband result directly (``residual_freq`` relative
    to channel centre, continuously evolving with absolute time, no
    window-hop artifact to begin with). Under the reading above, that
    means this module's output already satisfies BOTH ICD requirements
    by construction: there is no oversampling ramp to derotate because
    none was ever introduced, and for the zero-residual-frequency case
    (the convention's own reference point) this module's phase formula
    already evaluates to exactly zero at every sample, first-of-packet
    included -- trivially, since ``phase = 2*pi*residual_freq*t_local``
    is identically zero for all ``t_local`` when ``residual_freq = 0``.
  - What this does NOT resolve with certainty: whether CBF's receiver
    additionally expects literal per-heap phase discontinuities for a
    GENUINELY off-centre residual frequency (i.e. whether "zero phase at
    packet start" is meant as a NORMALIZATION reference point only, as
    reasoned above, or as a literal reset applied to every packet
    regardless of residual frequency, which would require this module to
    intentionally reintroduce a phase discontinuity at every heap
    boundary -- and would need CBF to reconstruct absolute phase
    continuity externally via heap_counter/channel_info, since the raw
    samples would no longer carry it). The former reading is far more
    physically sensible (a literal reset would discard real information
    -- frequency/delay -- that CBF's own delay-tracking and coherent
    beamforming need to recover across packets) and is what this module
    implements; flagged here, not silently assumed, so it can be
    confirmed against the real hardware/firmware behaviour (or CBF's own
    receive-side expectations) if a delay-tracking test ever shows a
    phase discontinuity at heap boundaries that this reasoning didn't
    predict.

NOISE — pre-generated tile bank
================================

Once the bank exists, per-tick cost is an index hash + a memcopy —
independent of channel count and, past a couple of threads, independent
of CPU budget too (benchmarked: ~8 cores clears the full 384-channel
budget with room to spare; even 1-2 would). This is NOT a free upgrade
over live per-tick generation — it is a deliberate fidelity/resource
tradeoff, and the tradeoff is REPEATS, not noise or CPU cost:

Filling the bank itself is plain ``numpy.random.Generator`` (see
``DirectSynthesisStreamer.__init__``) — a ONE-TIME, offline call at
construction, parallelized via a plain ``ThreadPoolExecutor`` over
independent ``SeedSequence.spawn()`` children. numpy's Generator
releases the GIL during generation, so this is genuine multi-core
speedup with zero custom numerical code. See CLAUDE.md's Benchmarking
section for the one-time construction-time budget this needs to fit
(target 10s, hard limit 30s) and measurements confirming it does.

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

``n_tiles`` and ``tile_n_samples`` ("tile width") are both configurable
— see ``DirectSynthesisStreamer``'s constructor. Bigger tiles or more of
them cost more one-time build time and memory (linear in both) and push
the first-repeat point out (roughly as sqrt(n_tiles), NOT linearly — no
memory-feasible size makes repeats rare over a full scan; it only delays
them a bit).

Noise never enters a delay pipeline here at all: receiver noise
originates locally at each station, after any signal-path delay would
apply, so applying station delay to it would be physically wrong.

PULSED (pulsar) sources
=======================

A pulsar is genuinely periodic, unlike noise: replaying one precomputed
period isn't a fidelity compromise, it's ground truth. So there is no
birthday-paradox tradeoff for pulsars — periodicity here is exact, to
the precision this simulator needs. (This design's rejected earlier
iterations, and why each was wrong, are documented in CLAUDE.md's
Pulsed sources section, not here.)

The current design generates the wideband, undispersed pulse train as
one real time series spanning the whole band (a shared "sky carrier" —
see below), applies the standard coherent-dispersion transfer function
(Lorimer & Kramer 2006, eq. 5.21) directly to its full complex FFT, then
channelizes (``_channelize_once``, a one-time, offline call — not a
hot-path FFT). Dispersing at full wideband FFT resolution before
channelizing correctly captures intra-channel dispersion smear as an
emergent property of the FFT resolution (a real low-frequency-radio
effect at SKA-Low frequencies — even DM=2 pc/cm^3 smears the dispersion
curve across tens of thousands of channel-widths at the bottom of the
band). Channelizing a real signal via FFT also inherently produces
genuinely complex per-channel content with real carrier phase, which
CBF's beamformer needs (it coherently combines stations by
phase-rotating already-channelized complex data) and which lets the
per-tick geometric-delay correction use tone's EXACT phase trick.

Verified against an external, peer-reviewed reference, not just internal
self-consistency: this module's dispersion constant (4148.808) matches
NANOGrav's PsrSigSim package's DM_K (1/2.41e-4 = 4149.38) to 0.014% —
the same standard literature constant, cross-validated independently
(PsrSigSim itself wasn't taken as a runtime dependency: its own package
is heavy and partially broken for this purpose — pulls in PINT, fitsio,
emcee, nestle, matplotlib just to import; its ``BasebandSignal.to_RF``/
``to_FilterBank`` conversions are unimplemented stubs — so its
``ISM._disperse_baseband`` physics was read directly and reimplemented
here instead).

WHY THE "SKY CARRIER" IS SHARED ACROSS STATIONS, NOT PER-STATION —
opposite of the rule for noise, easy to get backwards: every station in
a real array observes the literal same wavefront from the same source,
just arriving at a different time because of geometry — that's the
entire physical basis of interferometry. So the wideband pulse train's
random carrier is generated from a single fixed seed shared by every
station simulating this pulsar, never from ``station.station_id``.
Receiver noise is the opposite: independently seeded per station,
because each station's receiver is a physically separate noise source.

WHAT PULSED SOURCES DO NOT MODEL: pulse-to-pulse jitter, scintillation,
nulling, profile evolution with frequency, or realistic flux/SNR
calibration against the receiver noise floor — the last of these
matters specifically if the goal is testing whether PSS/PST can actually
detect the injected pulsar as a candidate, not just exercising
delay-tracking.

NUMBA — only where the PER-TICK budget actually requires it
==============================================================

numba is used only for genuinely per-tick, hot-path work — construction-
time work (noise-bank fill, the pulsar's wideband sky-carrier
generation) is plain/parallelized numpy instead, see their sections
above and CLAUDE.md's benchmarking notes:

    - ``synth_tone_channel``/``add_pulsar_tick``: fused per-(sample,
      channel) loops that avoid full-array temporaries. ``add_pulsar_tick``
      uses a phase-accumulator (NCO-style) recurrence rather than calling
      cos/sin per (sample, channel), which is significantly cheaper (see
      CLAUDE.md's bug #15).
    - ``_splitmix64_hash``: the only RNG-shaped code that's custom
      numba — used solely for picking each tick's noise tile index (a
      single integer hash + modulo, not a statistical distribution),
      because it must be callable per-tick with zero allocation from
      inside ``generate_next_tick``.
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
    MAX_NUM_CHANNELS,
    MIN_NUM_CHANNELS,
    NUM_CHANNELS,
    NUM_CHANNELS_STEP,
    OVERSAMPLING_DENOMINATOR,
    OVERSAMPLING_NUMERATOR,
    DelayFeed,
    DelayPolynomial,
    StationConfig,
    log,
)
from ska_low_station_beam_simulator.pulsar_catalog import load_pulsar_from_catalog

MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)

DEFAULT_N_TILES = 256

# Worker count for scipy.fft's multi-threaded FFT/IFFT calls in
# build_pulsar_template/_channelize_once (ONE-TIME construction, not
# per-tick). Benchmarked: no further benefit past 8 workers (bandwidth-
# bound at these array sizes) -- also matches this project's ~8-core/pod
# target, so this doesn't ask for more than a real deployment would have
# free during startup anyway.
PULSAR_FFT_WORKERS = 8

# Dispersion constant D (Lorimer & Kramer 2004/2006, eq. 5.1/5.21):
#   t_DM[s] = D * DM[pc cm^-3] / f[MHz]^2
# Cross-validated in tests/test_direct_synthesis.py against PsrSigSim's DM_K = 1/2.41e-4 = 4149.38
# (same standard constant, matches to within literature-typical precision).
DISPERSION_CONST_S_MHZ2_PER_DM = 4148.808

# The pulsar's intrinsic "sky carrier" seed -- shared by EVERY station
# simulating this pulsar (see module docstring). NEVER derive this from
# station.station_id.
DEFAULT_SKY_SEED = 0x5AB1E5EED


# ============================================================
# CORE KERNELS: splitmix64 hash (per-tick tile-index selection only),
# delay-polynomial eval, tone. No Box-Muller/statistical RNG lives in
# numba anymore -- see module docstring's NUMBA section.
# ============================================================


@njit(cache=True)
def _splitmix64_hash(seed, index):
    """Deterministic ``(seed, index) -> uint64`` hash, used ONLY to pick
    each tick's noise tile index (``generate_next_tick``:
    ``_splitmix64_hash(seed, tick_index) % n_tiles``) -- not a
    statistical distribution, just a well-distributed integer.

    Must be a PURE FUNCTION of ``index`` alone, with no state to track
    across calls: ``ScanRunner``/tests can and do call
    ``generate_next_tick`` with the SAME ``t`` twice and require
    byte-identical output (see ``test_tile_bank_determinism``), which
    rules out a stateful/sequential generator (a plain numpy
    ``Generator`` would advance and pick a different tile on the second
    call). Kept as a hand-rolled ``@njit`` function rather than an
    off-the-shelf numpy RNG for performance reasons benchmarked directly,
    not assumed -- see CLAUDE.md's bug #7 for the comparison against
    ``numpy.random.Philox`` (the one numpy BitGenerator that could
    otherwise provide this same "seek to an arbitrary index" property).

    :param seed: this source's own seed (e.g. a station's per-pol noise
        seed).
    :param index: the tick index to hash.
    :returns: a well-distributed ``uint64``, to be reduced mod
        ``n_tiles``.
    """
    x = (np.uint64(seed) ^ (np.uint64(index) * GOLDEN)) & MASK64
    x = (x + GOLDEN) & MASK64
    z = x
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & MASK64
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & MASK64
    z = z ^ (z >> np.uint64(31))
    return z


@njit(cache=True)
def eval_delay_poly_ns(coeffs, t_rel):
    """``t_rel`` MUST already be relative to the polynomial's own
    ``start_validity_sec`` (small magnitude) — same precision-safety
    requirement established for ``common.DelayPolynomial``.

    :param coeffs: polynomial coefficients, ascending order.
    :param t_rel: time relative to the polynomial's own
        ``start_validity_sec``.
    :returns: delay, in nanoseconds.
    """
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
    """Synthesizes ``n_samples`` of a tone in whichever channel ``freq_hz``
    maps to. Delay applied as continuous phase modulation — no ring
    buffer, no coarse/fine split. ``poly_t_rel_start`` and
    ``t_local_rel_start`` are both small-magnitude relative times (see
    module docstring's note on why raw epoch time breaks this).

    :param freq_hz: the tone's absolute frequency.
    :param amplitude: the tone's amplitude.
    :param base_freq_hz: channel 0's centre frequency.
    :param channel_width_hz: channel spacing.
    :param delay_coeffs: this source's delay polynomial coefficients.
    :param poly_t_rel_start: time relative to the polynomial's own
        ``start_validity_sec``, at sample 0.
    :param ypol_offset_ns: H-pol delay offset, in nanoseconds.
    :param is_h_pol: whether this call is for the H polarisation.
    :param t_local_rel_start: time relative to ``obs_time_ref``, at
        sample 0.
    :param sample_rate_per_channel: the per-channel output sample rate.
    :param n_samples: how many samples to synthesize.
    :returns: a ``(channel_index, samples)`` tuple.
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
# NOISE — tile bank fill, ONCE at construction. Plain numpy, not numba:
# this only runs once (unlike the per-tick replay it feeds), so there's
# no reason to avoid numpy's own Generator here. See module docstring.
# ============================================================


def fill_noise_bank(seed: int, std: float, n_tiles: int, tile_n_samples: int, num_channels: int) -> np.ndarray:
    """Fills a ``(n_tiles, tile_n_samples, num_channels)`` complex128 bank
    of independent complex Gaussian noise — statistically exact
    equivalent of "wideband noise, then FFT channelized" (DFT of i.i.d.
    Gaussian is i.i.d. Gaussian). NO delay applied — physically correct
    for receiver noise (see module docstring).

    Parallelized across a plain ``ThreadPoolExecutor`` over independent
    ``numpy.random.SeedSequence`` children — numpy's Generator releases
    the GIL during generation, so this genuinely uses multiple cores
    despite being plain Python/numpy, no numba required. Deterministic
    given ``(seed, n_tiles, tile_n_samples, num_channels)``: unlike the
    per-tick tile REPLAY (which must be a pure function of tick index so
    the same tick can be regenerated identically), the BANK ITSELF is
    built exactly once and never regenerated mid-scan, so it only needs
    to be reproducible run-to-run, not seekable to an arbitrary offset —
    a single seeded Generator stream is sufficient.

    Each worker writes directly into its slice of a single preallocated
    ``bank`` array, ONE TILE AT A TIME, bounding transient memory to a
    handful of tiles (a few tens of MB) regardless of the bank's total
    size (see CLAUDE.md's bug #16 for why this matters and what goes
    wrong without it). Plain assignment (``bank[i].real = arr``) is what
    makes this work: numpy's ``out=`` parameter would be the more obvious
    choice but requires a C-contiguous target, and ``.real``/``.imag``
    views of a complex array are strided.

    :param seed: seed for this bank's noise.
    :param std: per-sample standard deviation (both real and imaginary
        parts).
    :param n_tiles: number of tiles to generate.
    :param tile_n_samples: samples per tile, per channel.
    :param num_channels: channels per tile.
    :returns: the filled bank.
    """
    n_workers = max(1, min(n_tiles, os.cpu_count() or 8, 16))
    seed_seq = np.random.SeedSequence(seed)
    child_seqs = seed_seq.spawn(n_workers)
    base = n_tiles // n_workers
    remainder = n_tiles % n_workers
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(n_workers)]
    starts = np.cumsum([0] + chunk_sizes[:-1])

    bank = np.empty((n_tiles, tile_n_samples, num_channels), dtype=np.complex128)

    def _fill_range(child_seq, start, n_tiles_chunk):
        rng = np.random.default_rng(child_seq)
        for i in range(start, start + n_tiles_chunk):
            bank[i].real = rng.standard_normal((tile_n_samples, num_channels)) * std
            bank[i].imag = rng.standard_normal((tile_n_samples, num_channels)) * std

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        list(executor.map(_fill_range, child_seqs, starts, chunk_sizes))
    return bank


def bank_memory_bytes(n_tiles: int, tile_n_samples: int, num_channels: int, n_pols: int = 2) -> int:
    """Total resident memory for the noise bank across both pols.

    :param n_tiles: number of tiles in the bank.
    :param tile_n_samples: samples per tile, per channel.
    :param num_channels: channels per tile.
    :param n_pols: how many per-pol banks to account for.
    :returns: total bytes.
    """
    return n_tiles * tile_n_samples * num_channels * 16 * n_pols  # complex128 = 16 bytes


# ============================================================
# PULSAR — wideband generate + coherent dispersion + one-time
# channelization. See module docstring's PULSED (pulsar) sources section.
# ============================================================


def dispersion_delay_s(freq_hz: float, dm_pc_cm3: float) -> float:
    freq_mhz = freq_hz / 1e6
    return DISPERSION_CONST_S_MHZ2_PER_DM * dm_pc_cm3 / (freq_mhz**2)


def generate_wideband_pulse_train(seed, period_s, width_s, amplitude, wideband_rate, n_wide):
    """The shared 'sky carrier': one period of a real, wideband (spans
    the WHOLE band as a single time series), UNDISPERSED pulse train.
    Physically: incoherent broadband radio emission (noise-like),
    power-modulated by the pulsar's rotation -- amplitude = envelope(t)
    * a real Gaussian draw, same model PsrSigSim's ``_make_amp_pulses``
    uses.

    Plain vectorized numpy, not numba: this is a ONE-TIME construction
    call (see ``build_pulsar_template``) -- see CLAUDE.md's Benchmarking
    section for measurements.

    :param seed: seed for this pulsar's shared sky carrier (see module
        docstring for why it's shared across stations, not per-station).
    :param period_s: rotation period, in seconds.
    :param width_s: pulse profile FWHM, in seconds.
    :param amplitude: overall amplitude scale.
    :param wideband_rate: the sample rate of the wideband array, i.e.
        ``num_channels * channel_width_hz``.
    :param n_wide: total samples to generate (one period).
    :returns: the real-valued wideband pulse train, length ``n_wide``.
    """
    sigma = width_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # width_s = FWHM
    peak = period_s / 2.0
    t = np.arange(n_wide) / wideband_rate
    d = t - peak
    d = d - period_s * np.floor(d / period_s + 0.5)  # wrap into [-period/2, period/2)
    env = amplitude * np.exp(-0.5 * (d / sigma) ** 2)
    g = np.random.default_rng(seed).standard_normal(n_wide)
    return env * g


def _channelize_once(v: np.ndarray, num_channels: int, workers: int = PULSAR_FFT_WORKERS) -> np.ndarray:
    """One-shot FFT channelization of a complex 1D array of length
    ``n = k*num_channels`` into ``(k, num_channels)``, EXTERNAL
    ascending-channel-id order (channel c's center frequency is
    ``base_freq_hz + c*channel_width_hz`` for whatever ``base_freq_hz``
    the caller used to build ``v``'s frequency axis). A one-time,
    offline construction step, never the per-tick hot path.

    Uses ``scipy.fft`` (not ``numpy.fft``) with ``workers=`` -- this is
    the batched FFT along ``axis=-1`` over many independent
    ``(num_channels,)``-length rows, which is embarrassingly parallel
    across rows -- see CLAUDE.md's Benchmarking section for measurements.

    :param v: a complex 1D array, length a multiple of ``num_channels``.
    :param num_channels: channels to split ``v`` into.
    :param workers: ``scipy.fft`` worker thread count.
    :returns: ``(k, num_channels)`` complex spectra, external
        ascending-channel-id order.
    """
    n_out = v.shape[0] // num_channels
    v = v[: n_out * num_channels]
    windows = v.reshape(n_out, num_channels)
    spectra = scipy.fft.fft(windows, axis=-1, workers=workers)  # natural FFT bin order
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
    """ONE-TIME, offline construction (numpy + scipy.fft -- no per-tick
    budget applies here, so a real, multi-threaded FFT is fine, unlike
    the hot path). Generates the shared wideband pulse train, applies the
    coherent dispersion transfer function directly to its full complex
    FFT (capturing intra-channel smear as an emergent property of the
    full wideband frequency resolution), then channelizes via
    ``_channelize_once`` to produce genuinely complex per-channel content
    with real carrier phase.

    :param num_channels: channels to generate.
    :param channel_width_hz: channel spacing.
    :param base_freq_hz: channel 0's centre frequency.
    :param channel_output_rate: the per-channel output sample rate.
    :param period_s: rotation period, in seconds.
    :param width_s: pulse profile FWHM, in seconds.
    :param amplitude: overall amplitude scale.
    :param dm_pc_cm3: dispersion measure, in pc/cm^3.
    :param sky_seed: seed for the shared sky carrier (see module
        docstring).
    :returns: a ``(template, n_period_samples)`` tuple; ``template`` is
        ``(num_channels, n_period_samples)`` complex128, in this
        project's EXTERNAL ascending-channel-id order -- verified
        against a known-tone injection check in
        ``tests/test_direct_synthesis.py``, not assumed from reading the
        FFT-bin permutation logic.
    """
    wideband_rate = num_channels * channel_width_hz
    n_period_samples = int(round(period_s * channel_output_rate))
    n_wide = n_period_samples * num_channels  # exact multiple of num_channels,
    # required so _channelize_once's reshape doesn't drop a partial block.

    v = generate_wideband_pulse_train(sky_seed, period_s, width_s, amplitude, wideband_rate, n_wide)

    # Full complex FFT, NOT rfft -- this project's own convention (see
    # _channelize_once) treats negative fftfreq bins as meaningful,
    # independent channels. v_dispersed is genuinely complex after this
    # (dispersion breaks the real signal's Hermitian symmetry) --
    # expected, not a bug, and what gives the template real carrier
    # phase.
    V = scipy.fft.fft(v, workers=PULSAR_FFT_WORKERS)
    u = scipy.fft.fftfreq(n_wide, d=1.0 / wideband_rate)  # natural bin order, [-wideband_rate/2, wideband_rate/2)
    band_center_hz = base_freq_hz + wideband_rate / 2.0
    f_offset_mhz = u / 1e6
    f0_mhz = band_center_hz / 1e6
    # Lorimer & Kramer 2006, eq. 5.21 -- coherent dispersion transfer
    # function, cross-validated against PsrSigSim's ISM._disperse_baseband.
    H = np.exp(
        1j * 2 * np.pi * DISPERSION_CONST_S_MHZ2_PER_DM
        / ((f_offset_mhz + f0_mhz) * f0_mhz**2) * dm_pc_cm3 * f_offset_mhz**2
    )
    v_dispersed = scipy.fft.ifft(V * H, workers=PULSAR_FFT_WORKERS)

    channelized = _channelize_once(v_dispersed, num_channels)  # (n_period_samples, num_channels), external order
    template = np.ascontiguousarray(channelized.T)  # (num_channels, n_period_samples)
    return template, n_period_samples


@njit(parallel=True, cache=True)
def add_pulsar_tick(
    out, template, start_idx, n_period_samples,
    delay_coeffs, poly_t_rel_start, ypol_offset_ns, is_h_pol,
    base_freq_hz, channel_width_hz, channel_output_rate, n_samples, num_channels,
):
    """Adds this tick's contribution into ``out`` (``n_samples,
    num_channels`` complex128), reading the precomputed template
    circularly and applying the EXACT per-channel geometric-delay phase
    correction -- valid because the template is genuinely
    complex/narrowband per channel (see module docstring), the same
    reasoning that makes ``synth_tone_channel``'s delay-as-phase exact.
    Adds (not writes) so this composes with noise/tone in
    ``generate_next_tick``'s fixed order.

    Uses a phase-accumulator (NCO-style) recurrence rather than calling
    cos/sin per (sample, channel): ``phase(c) = phase(0) - c*dphase`` is
    linear in c at fixed sample, so each channel's rotation is the
    previous one times a single fixed per-sample step, computed via one
    complex multiply instead of two fresh transcendental calls per
    channel (see CLAUDE.md's bug #15 for why this matters).

    :param out: output buffer to accumulate into, ``(n_samples,
        num_channels)`` complex128.
    :param template: this pulsar's precomputed template, ``(num_channels,
        n_period_samples)`` complex128.
    :param start_idx: this tick's starting offset into ``template``'s
        period.
    :param n_period_samples: samples per pulsar period.
    :param delay_coeffs: this source's delay polynomial coefficients.
    :param poly_t_rel_start: time relative to the polynomial's own
        ``start_validity_sec``, at sample 0.
    :param ypol_offset_ns: H-pol delay offset, in nanoseconds.
    :param is_h_pol: whether this call is for the H polarisation.
    :param base_freq_hz: channel 0's centre frequency.
    :param channel_width_hz: channel spacing.
    :param channel_output_rate: the per-channel output sample rate.
    :param n_samples: samples in this tick.
    :param num_channels: channels in ``template``/``out``.
    """
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
    ):
        if not (
            MIN_NUM_CHANNELS <= num_channels <= MAX_NUM_CHANNELS
            and num_channels % NUM_CHANNELS_STEP == 0
        ):
            raise ValueError(
                f"num_channels={num_channels} is not a valid SPS beam "
                f"configuration -- per the ICD, the number of channels "
                f"assigned to a beam is configurable from "
                f"{MIN_NUM_CHANNELS} to {MAX_NUM_CHANNELS} in steps of "
                f"{NUM_CHANNELS_STEP} (384 channels * {CHANNEL_WIDTH_HZ:.0f}Hz "
                f"= 300MHz is the real maximum -- 448 was never a valid "
                f"configuration, see CLAUDE.md)."
            )

        for cfg in source_cfgs:
            if cfg["kind"] not in ("tone", "pulsed"):
                raise ValueError(
                    f"DirectSynthesisStreamer only supports kind in "
                    f"('tone', 'pulsed') in source_cfgs; got kind={cfg['kind']!r}."
                )
            if not isinstance(cfg.get("delay_feed"), DelayFeed):
                raise ValueError(
                    f"source_cfg kind={cfg['kind']!r} is missing a required "
                    f"'delay_feed' (a common.DelayFeed instance). There is no "
                    f"default/fallback delay for a source — a source with no "
                    f"real delay path would silently produce content that's "
                    f"trivially 'perfectly aligned', which could mask a real "
                    f"CBF delay-tracking bug instead of exercising it. Attach "
                    f"a DelayFeed to this source_cfg (e.g. via "
                    f"simulator.py's Tango attribute subscription, or "
                    f"directly in a test)."
                )
            if cfg["kind"] == "pulsed":
                # Two mutually exclusive ways to configure a pulsar: load
                # a pre-generated catalog entry by name (fast startup,
                # fixed parameters -- see pulsar_catalog.py) or supply
                # raw parameters to build a custom template at
                # construction (arbitrary parameters, pays the one-time
                # FFT construction cost -- see CLAUDE.md's Benchmarking
                # section for how tight that budget now is).
                param_keys = ("period_s", "width_s", "dm_pc_cm3")
                has_name = "pulsar_name" in cfg
                present_params = [k for k in param_keys if k in cfg]
                if has_name and present_params:
                    raise ValueError(
                        f"pulsed source_cfg has both 'pulsar_name' and "
                        f"{present_params} -- specify one or the other, "
                        f"not both: 'pulsar_name' loads a pre-generated "
                        f"catalog entry (see pulsar_catalog.py), "
                        f"'period_s'/'width_s'/'dm_pc_cm3' builds a custom "
                        f"template at construction."
                    )
                if not has_name and len(present_params) != len(param_keys):
                    missing = [k for k in param_keys if k not in cfg]
                    raise ValueError(
                        f"pulsed source_cfg needs either 'pulsar_name' "
                        f"(load a pre-generated catalog entry) or all of "
                        f"'period_s'/'width_s'/'dm_pc_cm3' (build a custom "
                        f"template at construction) -- got neither "
                        f"'pulsar_name' nor {missing}."
                    )

        self._pulsed_cfgs = [c for c in source_cfgs if c["kind"] == "pulsed"]
        if self._pulsed_cfgs and base_freq_hz <= 0:
            raise ValueError(
                f"pulsed source configured but base_freq_hz={base_freq_hz} -- "
                f"dispersion physics diverges as frequency -> 0 (see "
                f"dispersion_delay_s). The default (common.BASE_FREQ_HZ, "
                f"confirmed as 50.0 MHz, channel 64's centre frequency and "
                f"the lowest valid SKA-Low channel) is already real and "
                f"positive, so this only happens if base_freq_hz was "
                f"explicitly overridden to something invalid -- pass a "
                f"real, positive band-start frequency instead."
            )

        self.station = station
        self.num_channels = num_channels
        self.base_freq_hz = base_freq_hz
        self.channel_width_hz = channel_width_hz
        # The real, OVERSAMPLED per-channel output sample rate -- NOT
        # channel_width_hz (that would assume critical sampling). See
        # common.CHANNEL_OUTPUT_RATE_HZ's docstring for the ICD reference
        # and the derivation (32/27 oversampling factor -> 1080ns per
        # sample, not the naive 1280ns critical-sampling period).
        self.channel_output_rate = channel_width_hz * OVERSAMPLING_NUMERATOR / OVERSAMPLING_DENOMINATOR

        self._obs_time_ref = obs_time_ref

        # Each tone/pulsar carries its OWN delay feed (validated required,
        # above) — two sources at different sky directions genuinely have
        # different geometric delay. Each source also gets its own (poly
        # identity -> coeffs ndarray) cache so the coefficients array is
        # only rebuilt when that source's poly actually changes, not every
        # tick (see bug #13 in CLAUDE.md — per-tick allocation churn, not
        # compute, was the historical bottleneck here).
        self._tone_cfgs = [
            (c, c["delay_feed"], {}) for c in source_cfgs if c["kind"] == "tone"
        ]

        self._noise_cfg = noise_cfg
        self._noise_seed_v = noise_cfg["seed"] if noise_cfg else 0
        self._noise_seed_h = (noise_cfg["seed"] + 1_000_003) if noise_cfg else 0
        self._noise_std = noise_cfg["std"] if noise_cfg else 0.0

        # --- noise tile bank (see module docstring for the tradeoff) ---
        self.n_tiles = n_tiles
        self.tile_n_samples = tile_n_samples or self.tick_n_samples()
        self._banks: dict[str, np.ndarray] = {}
        if noise_cfg is not None:
            for pol, seed in (("V", self._noise_seed_v), ("H", self._noise_seed_h)):
                self._banks[pol] = fill_noise_bank(
                    seed, self._noise_std, self.n_tiles, self.tile_n_samples, self.num_channels
                )

        # --- pulsar templates: one (template, period_s, n_period_samples)
        # tuple per pulsed source cfg. Either LOADED from the catalog by
        # name (fast -- see pulsar_catalog.py) or BUILT here from raw
        # parameters, per source_cfg (validated mutually exclusive,
        # above). sky_seed is shared across all stations for the same
        # pulsar cfg by default (see module docstring) -- override only
        # for a deliberately DIFFERENT (uncorrelated) pulsar, never to
        # "vary" the same pulsar per station. ---
        self._pulsars = []
        for cfg in self._pulsed_cfgs:
            if "pulsar_name" in cfg:
                loaded = load_pulsar_from_catalog(
                    cfg["pulsar_name"],
                    self.num_channels,
                    self.base_freq_hz,
                    catalog_dir=cfg.get("catalog_dir"),
                )
                template = loaded["template"]
                amplitude = cfg.get("amplitude", 1.0)
                if amplitude != 1.0:
                    template = template * amplitude
                period_s = loaded["period_s"]
                n_period_samples = loaded["n_period_samples"]
            else:
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
                period_s = cfg["period_s"]
            self._pulsars.append((template, period_s, n_period_samples, cfg["delay_feed"], {}))

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
        """Per-channel output samples for one tick — ``HEAP_LEN`` by
        construction (``BLOCK_DURATION_S`` is defined for exactly this).

        :returns: per-channel samples for one tick.
        """
        return int(round(self.channel_output_rate * BLOCK_DURATION_S))

    def bank_memory_bytes(self) -> int:
        if not self._banks:
            return 0
        return bank_memory_bytes(self.n_tiles, self.tile_n_samples, self.num_channels, len(self._banks))

    @staticmethod
    def _coeffs_for(cache: dict, poly: DelayPolynomial) -> np.ndarray:
        """Returns ``poly``'s coefficients as a float64 array, only
        rebuilding it when ``poly`` (identity, not equality) actually
        changed since the last call — same per-source cache backing every
        tone/pulsar entry, so a source whose feed keeps returning the
        same poly object tick after tick (the common case) pays no
        per-tick allocation for this.

        :param cache: this source's own cache dict (one per tone/pulsar
            entry).
        :param poly: the delay polynomial currently in effect.
        :returns: ``poly.xypol_coeffs_ns`` as a float64 array.
        """
        if cache.get("poly") is not poly:
            cache["poly"] = poly
            cache["coeffs"] = np.asarray(poly.xypol_coeffs_ns, dtype=np.float64)
        return cache["coeffs"]

    def generate_next_tick(self, t: float, n_samples: int) -> dict[str, np.ndarray]:
        """See ``common.ScanRunner``, which sizes ``n_samples`` via
        ``tick_n_samples()`` so it lines up with ``HeapAccumulator``'s
        ``HEAP_LEN`` framing exactly.

        :param t: absolute epoch time of this tick's first sample.
        :param n_samples: PER-CHANNEL time samples for this tick (at
            ``channel_output_rate``).
        :returns: a dict mapping ``"V"``/``"H"`` to ``(n_samples,
            num_channels)`` complex arrays.
        """
        # t_local_rel_start (local clock for noise/tone/pulsar) must stay
        # small-magnitude — same precision requirement as everywhere else
        # in this codebase (see common.DelayPolynomial's docstring for why
        # raw epoch-scale time breaks this). Each source's own
        # poly_t_rel_start (relative to THAT source's own poly validity
        # window) is computed per-source below, once its feed is queried.
        t_local_rel_start = t - self._obs_time_ref

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
                out[:] = bank[tile_idx]
            else:
                out.fill(0)

            for template, period_s, n_period_samples, delay_feed, coeffs_cache in self._pulsars:
                poly = delay_feed.get(t)
                delay_coeffs = self._coeffs_for(coeffs_cache, poly)
                poly_t_rel_start = t - poly.start_validity_sec

                phase_in_period = t_local_rel_start % period_s
                start_idx = int(round(phase_in_period * self.channel_output_rate)) % n_period_samples
                add_pulsar_tick(
                    out, template, start_idx, n_period_samples,
                    delay_coeffs, poly_t_rel_start, poly.ypol_offset_ns, is_h_pol,
                    self.base_freq_hz, self.channel_width_hz, self.channel_output_rate,
                    n_samples, self.num_channels,
                )

            for cfg, delay_feed, coeffs_cache in self._tone_cfgs:
                poly = delay_feed.get(t)
                delay_coeffs = self._coeffs_for(coeffs_cache, poly)
                poly_t_rel_start = t - poly.start_validity_sec

                ch_idx, samples = synth_tone_channel(
                    cfg["freq_hz"],
                    cfg.get("amplitude", 1.0),
                    self.base_freq_hz,
                    self.channel_width_hz,
                    delay_coeffs,
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
