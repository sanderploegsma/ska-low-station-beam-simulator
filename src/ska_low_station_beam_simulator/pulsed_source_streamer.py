"""
Direct per-channel representation for PULSED sources — the piece
CLAUDE.md's "Immediate next steps" flags as still open ("derive a direct
per-channel representation for pulsed sources, to let wideband_streamer.py
be deleted entirely"). Grew out of a discussion about the tiled noise
bank: could the same "generate once, replay per tick" idea work for a
pulsar?

THE KEY DIFFERENCE FROM THE NOISE TILE BANK: a pulsar is genuinely
periodic. Replaying one precomputed period isn't a fidelity compromise
the way replaying a finite bank of "noise" tiles is (see
tiled_noise_streamer.py) — a real pulsar's profile repeats, to the
precision this simulator needs, exactly every rotation period. No
birthday-paradox tradeoff, no long-integration correctness caveat:
periodicity here is ground truth, not an artifact.

THIS MODULE WENT THROUGH THREE DESIGNS before arriving at the current
one, and the history matters for anyone tempted to "simplify" it back:

  v1 (WRONG): each channel = one constant DM delay, evaluated at that
  channel's center frequency, applied to a real-valued achromatic
  envelope; per-tick geometric delay applied as a first-order Taylor
  correction using a precomputed derivative. Two problems, both found by
  actually building and numerically checking, not by reasoning about it:

  v2 (fixed intra-channel smear, still broken for beamforming): a
  channel isn't one frequency, it's a ~781kHz-wide passband, and at
  SKA-Low frequencies even DM=2 pc/cm^3 smears the dispersion curve
  across tens of thousands of channel-widths at the bottom of the band —
  a real, well-known low-frequency effect, not a bug. v2 fixed this by
  averaging many shifted copies of the profile across each channel's own
  passband. But v2 was STILL real-valued (no carrier), which turns out
  to break something bigger: CBF's beamformer coherently combines
  stations by applying a complex phase rotation to already-channelized
  data — physically valid only because a real channelizer inherently
  produces complex baseband output with a genuine carrier phase tied to
  the channel's true center frequency. A real-valued, carrier-free
  representation has no phase for that rotation to act on, so v2's
  content could not be coherently beamformed across stations at all —
  tone doesn't have this problem (it has a genuine residual-frequency
  carrier by construction); v2's pulses did.

  v3 (this version): generate the wideband, undispersed pulse train as
  one real time series covering the whole band, apply the STANDARD
  coherent-dispersion transfer function (Lorimer & Kramer 2006, eq.
  5.21) to its real FFT, then channelize via
  wideband_streamer.WidebandChannelizer (reused here as a legitimate
  ONE-TIME, offline call — not a hot-path FFT, so it doesn't violate this
  codebase's "direct synthesis, no FFT per tick" philosophy). This fixes
  BOTH v1/v2 problems at once: intra-channel smear falls out correctly
  as an emergent property of dispersing at full wideband FFT resolution
  before channelizing (no averaging hack needed), and channelizing a
  real signal via FFT inherently produces genuinely complex per-channel
  content with real carrier phase — which is exactly what lets the
  per-tick geometric-delay correction use tone's EXACT phase trick
  instead of v1's Taylor approximation. The dispersion constant and
  transfer function here were cross-validated against the NANOGrav
  PsrSigSim package's ISM.disperse implementation (its DM_K = 1/2.41e-4
  = 4149.38, matching this module's own constant to within standard
  literature precision) — see this module's __main__.

WHY THE "SKY CARRIER" IS SHARED ACROSS STATIONS, NOT PER-STATION —
opposite of the rule for noise, easy to get backwards: every station in
a real array observes the literal SAME wavefront from the same source,
just arriving at a different time because of geometry. That's the whole
physical basis of interferometry — coherent combination only works
because it's genuinely the same signal, differentially delayed. So the
wideband pulse train's random "carrier" (physically: incoherent
broadband emission, amplitude-modulated by the pulsar's rotation — same
model PsrSigSim's _make_amp_pulses uses) is generated from a FIXED seed
shared by every station simulating this pulsar, never from
station.station_id. Receiver noise is the opposite: it must be
independently seeded per station, because each station's receiver is a
physically separate, uncorrelated noise source. Do not "fix" one to look
like the other.

WHAT THIS DOES NOT MODEL: pulse-to-pulse jitter, scintillation, nulling,
profile evolution with frequency, or realistic flux/SNR calibration
against the receiver noise floor — the last of these matters if the
downstream use case is testing whether PSS/PST can actually detect the
injected pulsar as a candidate, not just testing delay-tracking. See
CLAUDE.md.

REUSES direct_synthesis.py's tone/noise kernels and
wideband_streamer.WidebandChannelizer rather than re-deriving them, same
rationale as tiled_noise_streamer.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit, prange

from ska_low_station_beam_simulator.common import (
    BLOCK_DURATION_S,
    CHANNEL_WIDTH_HZ,
    DelayPolynomial,
    NUM_CHANNELS,
    StationConfig,
    fetch_delay_model_from_cbf,
    log,
)
from ska_low_station_beam_simulator.direct_synthesis import (
    _gaussian_pair,
    eval_delay_poly_ns,
    synth_noise_all_channels_into,
    synth_tone_channel,
)
from ska_low_station_beam_simulator.wideband_streamer import WidebandChannelizer

# Dispersion constant D (Lorimer & Kramer 2004/2006, eq. 5.1/5.21):
#   t_DM[s] = D * DM[pc cm^-3] / f[MHz]^2
# Cross-validated in __main__ against PsrSigSim's DM_K = 1/2.41e-4 = 4149.38
# (same standard constant, matches to within literature-typical precision).
DISPERSION_CONST_S_MHZ2_PER_DM = 4148.808

# common.py's BASE_FREQ_HZ is an explicit placeholder (0.0) pending ICD
# confirmation — dispersion delay diverges as f -> 0, so this module
# needs a real band-start frequency to be physically meaningful. Default
# to a plausible SKA-Low-band value; override via base_freq_hz for real use.
DEFAULT_PULSAR_BASE_FREQ_HZ = 50e6  # 50 MHz, illustrative SKA-Low low-band edge

# The pulsar's intrinsic "sky carrier" seed -- shared by EVERY station
# simulating this pulsar (see module docstring). NEVER derive this from
# station.station_id.
DEFAULT_SKY_SEED = 0x5AB1E5EED


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
    transfer function directly to its real FFT (capturing intra-channel
    smear as an emergent property of the full wideband frequency
    resolution), then channelizes via WidebandChannelizer to produce
    genuinely complex per-channel content with real carrier phase.

    Returns (template, n_period_samples): template is
    (num_channels, n_period_samples) complex64, in this project's
    EXTERNAL ascending-channel-id order (channel c's center frequency is
    base_freq_hz + c*channel_width_hz) -- verified against
    WidebandChannelizer's natural-bin-order output via a known-tone
    injection check in this module's __main__, not assumed from reading
    its permutation logic.
    """
    wideband_rate = num_channels * channel_width_hz
    n_period_samples = int(round(period_s * channel_output_rate))
    n_wide = n_period_samples * num_channels  # exact multiple of num_channels,
    # required so WidebandChannelizer (overlap=0) doesn't silently drop a
    # partial trailing block.

    v = generate_wideband_pulse_train(sky_seed, period_s, width_s, amplitude, wideband_rate, n_wide)

    # Full complex FFT, NOT rfft -- this project's own convention (see
    # WidebandChannelizer.channel_center_frequencies) treats negative
    # fftfreq bins as meaningful, independent channels (an IF/baseband
    # labeling convention where base_freq_hz is an additive offset, not
    # a strict real-signal Nyquist argument). Using rfft here originally
    # only covered half the intended band and shifted every channel by a
    # constant amount -- caught by the known-tone injection check below,
    # not by inspection. v_dispersed is genuinely complex after this
    # (dispersion breaks the real signal's Hermitian symmetry) -- that's
    # expected, not a bug, and is what gives the template real carrier
    # phase.
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

    channelizer = WidebandChannelizer(fft_len=num_channels, overlap=0)
    natural = channelizer.process(v_dispersed)  # (n_period_samples, num_channels), natural bin order
    template_by_natural = np.ascontiguousarray(natural.T)  # (num_channels, n_period_samples)
    template = np.empty_like(template_by_natural)
    template[channelizer._natural_to_external] = template_by_natural
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
    valid because the template is now genuinely complex/narrowband per
    channel (see module docstring), the same reasoning that makes
    synth_tone_channel's delay-as-phase exact. Adds (not writes) so this
    composes with noise/tone the same way tone composes on top of noise
    in the fixed DirectSynthesisStreamer.

    Uses a phase-accumulator (NCO-style) recurrence rather than calling
    cos/sin per (sample, channel) -- phase(c) = phase(0) - c*dphase is
    linear in c at fixed sample, so each channel's rotation is the
    previous one times a single fixed per-sample step, computed via one
    complex multiply instead of two fresh transcendental calls. An
    earlier version called cos/sin per channel directly and needed 4x
    the threads to clear budget for exactly the reason this codebase's
    noise kernel work already established: per-element transcendental
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


class PulsedSourceStreamer:
    """Same Streamer contract as DirectSynthesisStreamer, extended to
    handle kind='pulsed' alongside kind='tone', plus per-pol noise —
    the combination DirectSynthesisStreamer explicitly refuses. See
    module docstring for what's approximated and why."""

    def __init__(
        self,
        station: StationConfig,
        source_cfgs: list[dict],
        obs_time_ref: float,
        noise_cfg: Optional[dict] = None,
        num_channels: int = NUM_CHANNELS,
        base_freq_hz: float = DEFAULT_PULSAR_BASE_FREQ_HZ,
        channel_width_hz: float = CHANNEL_WIDTH_HZ,
    ):
        for cfg in source_cfgs:
            if cfg["kind"] not in ("tone", "pulsed"):
                raise ValueError(
                    f"PulsedSourceStreamer only supports kind in "
                    f"('tone', 'pulsed'); got kind={cfg['kind']!r}."
                )

        self.station = station
        self.num_channels = num_channels
        self.base_freq_hz = base_freq_hz
        self.channel_width_hz = channel_width_hz
        self.channel_output_rate = channel_width_hz

        self._obs_time_ref = obs_time_ref
        self._tone_cfgs = [c for c in source_cfgs if c["kind"] == "tone"]
        self._pulsed_cfgs = [c for c in source_cfgs if c["kind"] == "pulsed"]

        self._noise_cfg = noise_cfg
        self._noise_seed_v = noise_cfg["seed"] if noise_cfg else 0
        self._noise_seed_h = (noise_cfg["seed"] + 1_000_003) if noise_cfg else 0
        self._noise_std = noise_cfg["std"] if noise_cfg else 0.0

        self._current_poly: Optional[DelayPolynomial] = None
        self._delay_coeffs: Optional[np.ndarray] = None

        # One (template, period_s, n_period_samples) tuple per pulsed
        # source cfg, built ONCE here -- the whole point. sky_seed is
        # shared across all stations for the same pulsar cfg by default
        # (see module docstring) -- override only if you specifically
        # want two DIFFERENT (uncorrelated) pulsars, never to "vary"
        # the same pulsar per station.
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
        return int(round(self.channel_output_rate * BLOCK_DURATION_S))

    def _refresh_delay_poly_if_needed(self, t: float):
        if self._current_poly is None or t >= self._current_poly.valid_until:
            self._current_poly = fetch_delay_model_from_cbf(self.station.station_id, t)
            self._delay_coeffs = np.asarray(
                self._current_poly.xypol_coeffs_ns, dtype=np.float64
            )

    def generate_next_tick(self, t: float, n_samples: int) -> dict[str, np.ndarray]:
        self._refresh_delay_poly_if_needed(t)
        poly = self._current_poly

        t_local_rel_start = t - self._obs_time_ref
        poly_t_rel_start = t - poly.start_validity_sec

        results: dict[str, np.ndarray] = {}
        for pol, is_h_pol, noise_seed in (
            ("V", False, self._noise_seed_v),
            ("H", True, self._noise_seed_h),
        ):
            out = self._get_output_buffer(pol, n_samples)

            if self._noise_cfg is not None:
                sample_index_start = int(round(t_local_rel_start * self.channel_output_rate))
                synth_noise_all_channels_into(
                    out, noise_seed, self._noise_std, sample_index_start,
                    self.num_channels, n_samples,
                )
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
                    cfg["freq_hz"], cfg.get("amplitude", 1.0),
                    self.base_freq_hz, self.channel_width_hz,
                    self._delay_coeffs, poly_t_rel_start, poly.ypol_offset_ns,
                    is_h_pol, t_local_rel_start, self.channel_output_rate, n_samples,
                )
                if not (0 <= ch_idx < self.num_channels):
                    log.warning(
                        "tone freq_hz=%s maps to channel_idx=%d, outside the "
                        "configured [0, %d) channel range — skipping",
                        cfg["freq_hz"], ch_idx, self.num_channels,
                    )
                    continue
                out[:, ch_idx] += samples

            results[pol] = out

        return results


if __name__ == "__main__":
    # ============================================================
    # CORRECTNESS CHECKS
    # ============================================================
    def _fake_fetch(station_id, at_time):
        return DelayPolynomial(
            station_id=station_id, start_validity_sec=at_time, validity_period_sec=600.0,
            xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0], ypol_offset_ns=2.0,
        )

    globals()["fetch_delay_model_from_cbf"] = _fake_fetch

    # --- Channel-mapping ground-truth check: inject a KNOWN tone into
    # the wideband generation step (bypassing the pulse envelope/carrier
    # entirely) and confirm its power lands in the EXTERNAL channel index
    # this project's own convention predicts -- same style of check
    # direct_synthesis.py uses for synth_tone_channel, applied here to
    # verify the natural-to-external relabeling isn't silently wrong. ---
    NUM_CHANNELS_TEST = 32
    CHW = CHANNEL_WIDTH_HZ
    BASE_F = DEFAULT_PULSAR_BASE_FREQ_HZ
    wideband_rate = NUM_CHANNELS_TEST * CHW
    n_period_samples_test = 256
    n_wide_test = n_period_samples_test * NUM_CHANNELS_TEST
    test_channel = 7
    band_center_hz_test = BASE_F + wideband_rate / 2.0
    test_freq = BASE_F + test_channel * CHW + 0.15 * CHW  # off-center within the channel
    f_offset_test = test_freq - band_center_hz_test  # this project's own IF/baseband
    # convention (see WidebandChannelizer.channel_center_frequencies): fftfreq's
    # negative bins are meaningful distinct channels, not a real-signal-Nyquist
    # constraint -- so the injected test tone must be COMPLEX at the offset
    # frequency, not a real cosine (a real cosine's mirror-image negative
    # frequency component made an earlier version of this check silently
    # pass-or-fail on the wrong bin and is why the actual rfft/irfft bug
    # below wasn't obvious from reasoning alone).
    t_wide = np.arange(n_wide_test) / wideband_rate
    v_tone = np.exp(1j * 2 * np.pi * f_offset_test * t_wide)

    channelizer = WidebandChannelizer(fft_len=NUM_CHANNELS_TEST, overlap=0)
    natural = channelizer.process(v_tone)
    template_by_natural = natural.T
    relabeled = np.empty_like(template_by_natural)
    relabeled[channelizer._natural_to_external] = template_by_natural
    power_per_channel = np.mean(np.abs(relabeled) ** 2, axis=1)
    detected_channel = int(np.argmax(power_per_channel))
    print(
        f"tone injected at external channel {test_channel} (freq={test_freq/1e6:.4f} MHz) "
        f"-> detected at external channel {detected_channel}: "
        f"{'OK' if detected_channel == test_channel else 'MISMATCH -- relabeling logic is wrong'}"
    )
    assert detected_channel == test_channel

    # --- dispersion constant cross-check against PsrSigSim's DM_K ---
    psrsigsim_dm_k = 1.0 / 2.41e-4
    rel_diff = abs(DISPERSION_CONST_S_MHZ2_PER_DM - psrsigsim_dm_k) / psrsigsim_dm_k
    print(
        f"dispersion constant: this module={DISPERSION_CONST_S_MHZ2_PER_DM}  "
        f"PsrSigSim DM_K={psrsigsim_dm_k:.3f}  relative diff={rel_diff*100:.3f}% "
        f"(expected: small, standard literature-precision variation)"
    )

    # --- full pipeline: build a real pulsar template, check determinism ---
    NUM_CHANNELS_TEST2 = 448
    DM_TEST = 2.0
    PERIOD_S = 0.1
    WIDTH_S = 0.005

    station = StationConfig(station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1)
    streamer = PulsedSourceStreamer(
        station=station,
        source_cfgs=[{"kind": "pulsed", "period_s": PERIOD_S, "width_s": WIDTH_S, "amplitude": 1.0, "dm_pc_cm3": DM_TEST}],
        obs_time_ref=1_800_000_000.0,
        num_channels=NUM_CHANNELS_TEST2,
    )
    n_samples = streamer.tick_n_samples()
    tick_dt = n_samples / streamer.channel_output_rate
    obs_time = 1_800_000_000.0

    r1 = streamer.generate_next_tick(obs_time, n_samples)["V"].copy()
    r2 = streamer.generate_next_tick(obs_time, n_samples)["V"].copy()
    assert np.array_equal(r1, r2), "same t must give identical content"
    print("determinism/seekability: OK")
    print(f"content is complex (not real-only): max |imag part| = {np.max(np.abs(r1.imag)):.4f}")

    # --- cross-station coherence check: two stations, same pulsar, same
    # tick, but different geometric delay via their own DelayPolynomial
    # -- confirm applying each station's OWN delay-compensating phase
    # rotation brings them into close alignment (what a beamformer relies
    # on), which v1/v2's real-only content could not do. ---
    def _fake_fetch_b(station_id, at_time):
        return DelayPolynomial(
            station_id=station_id, start_validity_sec=at_time, validity_period_sec=600.0,
            xypol_coeffs_ns=[300.0, 0.002, 0.0, 0.0, 0.0, 0.0], ypol_offset_ns=1.0,
        )

    station_b = StationConfig(station_id=2, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1)
    streamer_b = PulsedSourceStreamer(
        station=station_b,
        source_cfgs=[{"kind": "pulsed", "period_s": PERIOD_S, "width_s": WIDTH_S, "amplitude": 1.0, "dm_pc_cm3": DM_TEST}],
        obs_time_ref=1_800_000_000.0,
        num_channels=NUM_CHANNELS_TEST2,
    )
    globals()["fetch_delay_model_from_cbf"] = _fake_fetch_b
    streamer_b._refresh_delay_poly_if_needed(obs_time)
    globals()["fetch_delay_model_from_cbf"] = _fake_fetch
    out_b = streamer_b.generate_next_tick(obs_time, n_samples)["V"]
    out_a = streamer.generate_next_tick(obs_time, n_samples)["V"]

    test_ch = 200
    tau_a_ns = eval_delay_poly_ns(streamer._delay_coeffs, obs_time - streamer._current_poly.start_validity_sec)
    tau_b_ns = eval_delay_poly_ns(streamer_b._delay_coeffs, obs_time - streamer_b._current_poly.start_validity_sec)
    f_c = streamer.base_freq_hz + test_ch * streamer.channel_width_hz
    correction_a = np.exp(1j * 2 * np.pi * f_c * tau_a_ns * 1e-9)
    correction_b = np.exp(1j * 2 * np.pi * f_c * tau_b_ns * 1e-9)
    aligned_a = out_a[:, test_ch] * correction_a
    aligned_b = out_b[:, test_ch] * correction_b
    corr = np.abs(np.vdot(aligned_a, aligned_b)) / np.sqrt(np.vdot(aligned_a, aligned_a).real * np.vdot(aligned_b, aligned_b).real)
    print(
        f"cross-station coherence after delay-compensating phase rotation: "
        f"normalized correlation={corr:.4f} (expect close to 1.0 -- this is what "
        f"v1/v2's real-only content could NOT achieve)"
    )

    print("\nAll checks completed.")
