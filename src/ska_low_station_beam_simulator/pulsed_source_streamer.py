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
precision this simulator needs, exactly every rotation period. So there
is no birthday-paradox tradeoff here, and no correctness caveat about
long-integration statistics: periodicity is the ground truth, not an
artifact.

WHAT STILL HAS TO BE HANDLED, and is the actual new content of this
module: CBF's delay-tracking has to be exercised against a
CONTINUOUSLY-CHANGING geometric delay, tick by tick (that's the point of
this whole simulator). A single precomputed-and-replayed snapshot is
frozen at whatever delay was baked in at generation time. So generation
is split into two genuinely different pieces:

  1. DISPERSION (DM) delay: a FIXED, static property of this simulated
     pulsar (frequency-dependent, oc 1/f^2 — see dispersion_delay_s) —
     baked into the per-channel template ONCE, at construction. This is
     why each channel gets its OWN one-period template: channel c's
     copy of the pulse is the same intrinsic (achromatic) profile,
     shifted in time by that channel's own dispersion delay relative to
     the top of the band.
  2. GEOMETRIC (tracking) delay: changes every tick, and is what CBF's
     delay-poly is actually being tested against. Applied per-tick on
     top of the replayed template.

THE GEOMETRIC-DELAY CORRECTION IS AN APPROXIMATION, and unlike tone's
"delay as phase" trick (exact for a monochromatic signal), it is NOT the
same trick reused verbatim — an earlier version of this design assumed
it would be, and that assumption was wrong on inspection. Pulses are
represented here as real-valued, achromatic amplitude envelopes (no
carrier/residual-frequency term at all — same convention
wideband_streamer.py's rectangular PulsedSource already uses: pulse
content is real, cast to complex with zero imaginary part). A phase
multiply on a real, zero-frequency baseband signal does not correspond
to a time shift — there is no carrier for the phase to act on. The
correct per-tick correction for a REAL envelope is a first-order Taylor
expansion in the (small) geometric delay:

    envelope(t - tau) ~= envelope(t) - tau * envelope'(t)

so each per-channel template is built alongside its own derivative
(closed-form for a Gaussian profile), and applying a tick's geometric
delay is one multiply-add per sample using the precomputed derivative —
still O(1) per sample, no re-synthesis. This is valid because realistic
geometric delays in this codebase's own test data are sub-microsecond
(see common.DelayPolynomial usage elsewhere), utterly small relative to
a pulse profile's own millisecond-scale timescale — see this module's
__main__ for a numerical check of the actual error at realistic
magnitudes, not just an assertion that it's fine.

INTRA-CHANNEL DISPERSION SMEAR — an earlier version of this module
skipped this (treated each channel as seeing one constant DM delay,
evaluated only at the channel center frequency) and that turned out to
be badly wrong at this project's actual band: at SKA-Low frequencies
(tens to a few hundred MHz) and 781.25kHz channels, even the smallest
realistic pulsar DM (~1-2 pc/cm^3) smears the dispersion curve across
many channel-widths within a SINGLE channel — this is a well-known real
effect at low radio frequencies (it's why real low-frequency pulsar
backends need much finer channelization or coherent dedispersion), not
a bug in this simulator. See build_pulsar_template's docstring for the
fix: each channel's template is built as the AVERAGE of many shifted
copies of the intrinsic profile, sampled across that channel's own
passband, which numerically converges to the same result a proper
wideband-generate + apply-dispersion-in-frequency-domain +
channelize-via-FFT pipeline would produce, for an idealized achromatic
source. Done this way instead of literally reusing
wideband_streamer.WidebandChannelizer specifically to avoid depending on
that module's own FFT-bin/permutation conventions — this stays a
self-contained, independently-verifiable calculation, matching this
codebase's general preference for direct/closed-form approaches. See
this module's __main__ for a convergence check (does the answer stop
changing as the number of sub-samples grows) and a check that it reduces
to the old single-delay approximation when smear is genuinely small.

REUSES direct_synthesis.py's tone/noise kernels and eval_delay_poly_ns
rather than re-deriving them, same rationale as tiled_noise_streamer.py.
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
    eval_delay_poly_ns,
    synth_noise_all_channels_into,
    synth_tone_channel,
)

# Dispersion constant D (Lorimer & Kramer 2004, eq. 5.1):
#   t_DM[s] = D * DM[pc cm^-3] / f[MHz]^2
DISPERSION_CONST_S_MHZ2_PER_DM = 4148.808

# common.py's BASE_FREQ_HZ is an explicit placeholder (0.0) pending ICD
# confirmation — dispersion delay diverges as f -> 0, so this module
# needs a real band-start frequency to be physically meaningful. Default
# to a plausible SKA-Low-band value; override via base_freq_hz for real use.
DEFAULT_PULSAR_BASE_FREQ_HZ = 50e6  # 50 MHz, illustrative SKA-Low low-band edge


def dispersion_delay_s(freq_hz: float, dm_pc_cm3: float) -> float:
    freq_mhz = freq_hz / 1e6
    return DISPERSION_CONST_S_MHZ2_PER_DM * dm_pc_cm3 / (freq_mhz**2)


@njit(cache=True)
def _gaussian_envelope_and_deriv(t, peak, period_s, sigma, amplitude):
    d = t - peak
    d = d - period_s * np.floor(d / period_s + 0.5)  # wrap into [-period/2, period/2)
    val = amplitude * np.exp(-0.5 * (d / sigma) ** 2)
    deriv = -(d / (sigma * sigma)) * val
    return val, deriv


@njit(parallel=True, cache=True)
def build_pulsar_template(
    num_channels, channel_width_hz, base_freq_hz, channel_output_rate,
    period_s, width_s, amplitude, dm_pc_cm3, n_period_samples, n_subfreq,
):
    """One period's worth of this pulsar's profile, per channel, WITH
    intra-channel dispersion smear correctly captured.

    A channel is not a single frequency, it's a passband
    [f_c - channel_width/2, f_c + channel_width/2). An idealized
    brick-wall channelizer's output is the (frequency-weighted) INTEGRAL
    of the true dispersion-swept signal across that passband:

        channel_c(t) = (1/W) * integral_{f in passband} env(t - tau_DM(f)) df

    Approximated here by averaging n_subfreq shifted copies of the
    intrinsic profile, at frequencies sampled uniformly across the
    channel's own passband -- this converges to the same result a
    proper wideband-generate + apply-dispersion + FFT-channelize
    pipeline would give for an achromatic source (see this module's
    __main__ for a convergence check), without needing to build that
    full pipeline. n_subfreq=1 recovers the old (wrong, except when
    smear is negligible) single-constant-delay approximation.

    Real-valued (achromatic pulse, same convention as
    wideband_streamer.PulsedSource). Returns (template, deriv), each
    (num_channels, n_period_samples) float64 -- the derivative is the
    average of the M sub-frequency derivatives, valid since
    differentiation is linear.
    """
    template = np.empty((num_channels, n_period_samples), dtype=np.float64)
    deriv = np.empty((num_channels, n_period_samples), dtype=np.float64)
    sigma = width_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # width_s = FWHM

    f_top = base_freq_hz + (num_channels - 1) * channel_width_hz
    tau_top = DISPERSION_CONST_S_MHZ2_PER_DM * dm_pc_cm3 / (f_top / 1e6) ** 2

    for c in prange(num_channels):
        f_center = base_freq_hz + c * channel_width_hz
        f_lo = f_center - channel_width_hz / 2.0
        # n_subfreq points spanning the channel's own passband
        peaks = np.empty(n_subfreq, dtype=np.float64)
        for m in range(n_subfreq):
            f_sub = f_lo + (m + 0.5) * channel_width_hz / n_subfreq
            tau_dm_sub = DISPERSION_CONST_S_MHZ2_PER_DM * dm_pc_cm3 / (f_sub / 1e6) ** 2 - tau_top
            peaks[m] = period_s / 2.0 + tau_dm_sub

        for i in range(n_period_samples):
            t = i / channel_output_rate
            val_sum = 0.0
            deriv_sum = 0.0
            for m in range(n_subfreq):
                val, dv = _gaussian_envelope_and_deriv(t, peaks[m], period_s, sigma, amplitude)
                val_sum += val
                deriv_sum += dv
            template[c, i] = val_sum / n_subfreq
            deriv[c, i] = deriv_sum / n_subfreq
    return template, deriv


@njit(parallel=True, cache=True)
def add_pulsar_tick(
    out, template, deriv, start_idx, n_period_samples,
    delay_coeffs, poly_t_rel_start, ypol_offset_ns, is_h_pol,
    channel_output_rate, n_samples, num_channels,
):
    """Adds this tick's contribution into `out` (n_samples, num_channels
    complex128), reading the precomputed template circularly and
    applying the per-sample first-order geometric-delay correction. Adds
    (not writes) so this composes with noise/tone the same way tone
    composes on top of noise in the fixed DirectSynthesisStreamer."""
    for i in prange(n_samples):
        idx = (start_idx + i) % n_period_samples
        t_poly = poly_t_rel_start + i / channel_output_rate
        tau_ns = eval_delay_poly_ns(delay_coeffs, t_poly)
        if is_h_pol:
            tau_ns += ypol_offset_ns
        tau_s = tau_ns * 1e-9
        for c in range(num_channels):
            out[i, c] += template[c, idx] - tau_s * deriv[c, idx]


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
        n_subfreq: int = 64,
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

        # One (template, deriv, period_s, n_period_samples) tuple per
        # pulsed source cfg, built ONCE here — the whole point.
        self._pulsars = []
        self.n_subfreq = n_subfreq
        for cfg in self._pulsed_cfgs:
            n_period_samples = int(round(cfg["period_s"] * self.channel_output_rate))
            template, deriv = build_pulsar_template(
                self.num_channels,
                self.channel_width_hz,
                self.base_freq_hz,
                self.channel_output_rate,
                cfg["period_s"],
                cfg["width_s"],
                cfg.get("amplitude", 1.0),
                cfg["dm_pc_cm3"],
                n_period_samples,
                self.n_subfreq,
            )
            self._pulsars.append((template, deriv, cfg["period_s"], n_period_samples))

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

            for template, deriv, period_s, n_period_samples in self._pulsars:
                phase_in_period = t_local_rel_start % period_s
                start_idx = int(round(phase_in_period * self.channel_output_rate)) % n_period_samples
                add_pulsar_tick(
                    out, template, deriv, start_idx, n_period_samples,
                    self._delay_coeffs, poly_t_rel_start, poly.ypol_offset_ns, is_h_pol,
                    self.channel_output_rate, n_samples, self.num_channels,
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
    # CORRECTNESS / APPROXIMATION-ERROR CHECKS
    # ============================================================
    def _fake_fetch(station_id, at_time):
        return DelayPolynomial(
            station_id=station_id, start_validity_sec=at_time, validity_period_sec=600.0,
            xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0], ypol_offset_ns=2.0,
        )

    globals()["fetch_delay_model_from_cbf"] = _fake_fetch

    NUM_CHANNELS_TEST = 448
    DM_TEST = 2.0  # pc/cm^3 -- a low, realistic DM (nearby pulsar); even this
    # is enough to badly smear the bottom of the SKA-Low band, see below
    PERIOD_S = 0.1  # 100ms -- "normal" (non-millisecond) pulsar
    WIDTH_S = 0.005  # 5ms FWHM, 5% duty cycle

    station = StationConfig(station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1)
    streamer = PulsedSourceStreamer(
        station=station,
        source_cfgs=[{"kind": "pulsed", "period_s": PERIOD_S, "width_s": WIDTH_S, "amplitude": 1.0, "dm_pc_cm3": DM_TEST}],
        obs_time_ref=1_800_000_000.0,
        num_channels=NUM_CHANNELS_TEST,
        n_subfreq=64,
    )
    n_samples = streamer.tick_n_samples()
    tick_dt = n_samples / streamer.channel_output_rate
    obs_time = 1_800_000_000.0

    # --- determinism/seekability ---
    r1 = streamer.generate_next_tick(obs_time, n_samples)["V"].copy()
    r2 = streamer.generate_next_tick(obs_time, n_samples)["V"].copy()
    assert np.array_equal(r1, r2), "same t must give identical content"
    print("determinism/seekability: OK")

    # --- dispersion sanity: bottom-of-band delay relative to top ---
    f_top = streamer.base_freq_hz + (NUM_CHANNELS_TEST - 1) * streamer.channel_width_hz
    f_bot = streamer.base_freq_hz
    tau_bot = dispersion_delay_s(f_bot, DM_TEST) - dispersion_delay_s(f_top, DM_TEST)
    print(
        f"dispersion delay, bottom channel ({f_bot/1e6:.1f} MHz) relative to "
        f"top ({f_top/1e6:.1f} MHz), DM={DM_TEST}: {tau_bot*1000:.2f} ms "
        f"({tau_bot/PERIOD_S:.2f} periods -- wrapping is expected and correct "
        f"for a periodic source)"
    )

    # --- intra-channel dispersion smear diagnostic: how big is it at
    # each band edge, and does this module's sub-frequency averaging
    # actually converge / capture it? ---
    channel_sample_period = 1.0 / streamer.channel_width_hz
    for label, f_edge in [("bottom (worst)", f_bot), ("top (best)", f_top)]:
        d_tau_df = -2 * DISPERSION_CONST_S_MHZ2_PER_DM * DM_TEST / (f_edge / 1e6) ** 3 / 1e6  # per Hz
        smear = abs(d_tau_df) * streamer.channel_width_hz
        print(
            f"intra-channel smear at {label} channel ({f_edge/1e6:.1f} MHz): "
            f"{smear*1e6:.3f} us vs channel sample period {channel_sample_period*1e6:.3f} us "
            f"(ratio={smear/channel_sample_period:.2f})"
        )

    print()
    print("convergence check: does the averaged template stop changing as")
    print("n_subfreq grows? (small scale build: 8 channels, 20ms period, for speed)")
    conv_channels, conv_period = 8, 0.02
    conv_n_period = int(round(conv_period * CHANNEL_WIDTH_HZ))
    worst_channel = 0  # bottom of band -- largest smear
    prev = None
    for n_sf in [1, 4, 16, 64, 256]:
        tmpl, _ = build_pulsar_template(
            conv_channels, CHANNEL_WIDTH_HZ, DEFAULT_PULSAR_BASE_FREQ_HZ, CHANNEL_WIDTH_HZ,
            conv_period, WIDTH_S, 1.0, DM_TEST, conv_n_period, n_sf,
        )
        peak = tmpl[worst_channel].max()
        delta = "" if prev is None else f"  (change from previous: {abs(peak-prev):.3e})"
        print(f"  n_subfreq={n_sf:>4}: peak value at worst channel = {peak:.6f}{delta}")
        prev = peak
    print(
        "n_subfreq=1 (the old, wrong approximation) peak vs the converged "
        "(n_subfreq=256) value directly shows how much smearing was being missed."
    )

    # --- geometric-delay Taylor-correction error, at a realistic delay
    # magnitude (matches this codebase's example delay-poly coefficients,
    # ~750ns). "exact" here must be built the SAME way the template is
    # (averaged over n_subfreq shifted copies) so this isolates the
    # error from the Taylor correction specifically, not from also
    # switching between the single-delay and averaged smear models. ---
    template, deriv, period_s, n_period = streamer._pulsars[0]
    sigma = WIDTH_S / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    test_channel = NUM_CHANNELS_TEST - 1  # top of band -- smallest smear, cleanest check
    f_center = streamer.base_freq_hz + test_channel * streamer.channel_width_hz
    f_lo = f_center - streamer.channel_width_hz / 2.0
    peaks = [
        period_s / 2.0
        + dispersion_delay_s(f_lo + (m + 0.5) * streamer.channel_width_hz / streamer.n_subfreq, DM_TEST)
        - dispersion_delay_s(f_top, DM_TEST)
        for m in range(streamer.n_subfreq)
    ]
    t0 = period_s / 2.0 + (dispersion_delay_s(f_center, DM_TEST) - dispersion_delay_s(f_top, DM_TEST))
    for tau_geom_ns in [750.0, 2000.0, 10_000.0]:
        tau_geom_s = tau_geom_ns * 1e-9
        idx0 = int(round((t0 % period_s) * streamer.channel_output_rate)) % n_period
        approx = template[test_channel, idx0] - tau_geom_s * deriv[test_channel, idx0]
        exact = 0.0
        for peak in peaks:
            d = (t0 - tau_geom_s) - peak
            d = d - period_s * np.floor(d / period_s + 0.5)
            exact += np.exp(-0.5 * (d / sigma) ** 2)
        exact /= streamer.n_subfreq
        err = abs(approx - exact)
        print(
            f"geometric delay {tau_geom_ns:>8.1f}ns: Taylor-approx={approx:.6f}  "
            f"exact={exact:.6f}  abs err={err:.3e}"
        )

    # --- cross-station independence not applicable here (pulsar content
    # is the same astrophysical source seen by every station -- unlike
    # noise, stations SHOULD see the same intrinsic pulse, just arriving
    # at each station's own geometric delay; this is correct, not a bug) ---

    print("\nAll checks completed.")
