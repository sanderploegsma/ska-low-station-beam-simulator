"""
Direct per-channel synthesis, bypassing wideband time-domain generation +
FFT channelization entirely — for TONE and NOISE sources only. Pulsed
sources are explicitly parked (see note at bottom).

WHY THIS EXISTS:
The wideband+FFT approach (wideband_streamer.py) generates a wideband
time-domain signal across the full instantaneous bandwidth, then
FFT-channelizes it. Cost analysis:
    - Per-tick work scales as O(NUM_CHANNELS) for generation, worse
      than O(NUM_CHANNELS) for the FFT (N log N).
    - The per-tick TIME BUDGET is fixed (BLOCK_DURATION_S =
      HEAP_LEN/CHANNEL_WIDTH_HZ), independent of channel count.
    - Scaling from 96 channels (75 MHz) to 448 channels (350 MHz, full
      SKA-Low band) is a ~4.67x increase in work against an unchanged
      budget — on top of an architecture already sitting at ~1.1-1.6x
      the budget at 96 channels. Not viable as-is.

THE ALTERNATIVE, per source type:
    - TONE: spectrally sparse — after channelization it lives almost
      entirely in ONE channel. Synthesize that channel's already-
      channelized time series directly via a closed-form complex
      exponential at the tone's frequency residual relative to that
      channel's center. Cost is O(1) per tone — INDEPENDENT of total
      channel count. Delay becomes a continuous phase term applied
      directly in the exponent — no ring buffer, no coarse/fine
      integer-sample split needed at all. This is exact for a truly
      monochromatic tone, not an approximation: the integer/fractional
      sample delay distinction is an artifact of discretized time-domain
      representation, and direct phase modulation sidesteps it entirely.
    - NOISE: the DFT of i.i.d. complex Gaussian noise is itself i.i.d.
      complex Gaussian (a unitary transform preserves this) — so
      generating independent Gaussian samples directly at
      (heap_sample, channel) resolution reproduces the exact statistics
      of "wideband noise, then channelized," without ever computing an
      FFT. Cost is O(HEAP_LEN x NUM_CHANNELS) — same order as the
      wideband approach's generation ALONE, with channelization's cost
      (~65-75% of the total tick there) removed entirely.
    - Also fixes a real physics bug present in wideband_streamer.py:
      station (receiver) noise gets delay-corrected identically to the
      sky signal there, which is wrong — receiver noise originates
      locally at each station, after any signal-path delay would apply.
      Direct per-channel noise synthesis never touches the delay
      pipeline at all, so this is fixed as a side effect, not a
      separate patch.
    - PULSED sources: PARKED. Broadband by nature, so the "lives in one
      bin" shortcut doesn't apply — there's likely a cheaper closed-form
      per-channel representation available, but it isn't derived here.
      Don't assume it's a trivial extension of the tone case. Use
      wideband_streamer.StationStreamer for pulsed sources.

NUMBA: benchmarked directly against a plain numpy/Philox implementation
before committing to it here (see CLAUDE.md's benchmarking notes) —
unlike some of the wideband path's optimizations, this one earns its
complexity:
    - Noise (the dominant cost at high channel counts): numba+prange
      beats numpy's vectorized Generator(Philox) by ~3x at 8-10 threads
      on a 10-core Apple Silicon machine (2.3ms vs 7.3ms at 448 channels,
      2048 samples), because numpy's Generator has no built-in
      multi-threading — numba's the only way to actually parallelize
      this stage. Kept.
    - Tone: numba's fused loop (14.5us) still beats a vectorized numpy
      version (25.1us) by avoiding several full-array temporaries
      (t_local, t_poly, tau_ns, phase) — a smaller win, but tone is
      already so cheap it doesn't matter for the budget either way.
      Kept for consistency with the noise kernel and because it isn't
      worse.
Same underlying reason Philox isn't used here as noise's RNG (see
wideband_streamer.py's docstring): NOT supported in numba nopython mode.
Uses an independent splitmix64-style hash + Box-Muller implementation
instead — a separate copy from wideband_streamer.py's, deliberately (see
that module's docstring on why the two backends don't share code).
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


@njit(parallel=True, cache=True)
def synth_noise_all_channels(seed, std, sample_index_start, num_channels, n_samples):
    """
    Independent complex Gaussian directly at (n_samples, num_channels)
    resolution — statistically exact equivalent of "wideband noise, then
    FFT channelized" (DFT of i.i.d. Gaussian is i.i.d. Gaussian). NO
    delay applied — physically correct for receiver noise, and a fix
    for the bug described in this module's docstring.
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


# ============================================================
# DIRECT SYNTHESIS STREAMER
#
# SCOPE: tone + per-pol station (receiver) noise only. Pulsed sources are
# explicitly PARKED — a source_cfgs entry with kind='pulsed' raises here
# rather than being silently dropped or mishandled; use
# wideband_streamer.StationStreamer for pulsed sources.
#
# OUTPUT CONTRACT: generate_next_tick returns dict[pol] -> (n_samples,
# num_channels) complex array, exactly like WidebandChannelizer's output,
# so it plugs into common.HeapAccumulator/SpsPacketizer completely
# unchanged. Columns are already in EXTERNAL ascending-frequency
# channel_id order (synth_tone_channel computes channel_idx directly from
# frequency) — there is no FFT natural-bin-order permutation to undo
# here, unlike the wideband path, so channel_id_map is just identity.
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
    ):
        for cfg in source_cfgs:
            if cfg["kind"] != "tone":
                raise ValueError(
                    f"DirectSynthesisStreamer only supports kind='tone' in "
                    f"source_cfgs — pulsed sources are explicitly parked (no "
                    f"per-channel closed form derived yet, see this module's "
                    f"docstring); got kind={cfg['kind']!r}. Use "
                    f"wideband_streamer.StationStreamer for pulsed sources."
                )

        self.station = station
        self.num_channels = num_channels
        self.base_freq_hz = base_freq_hz
        self.channel_width_hz = channel_width_hz
        # Critically sampled per-channel output rate — matches
        # WidebandChannelizer's channel_center_frequencies spacing when
        # OVERLAP=0 (the wideband path's only benchmarked/verified config).
        self.channel_output_rate = channel_width_hz

        self._obs_time_ref = obs_time_ref
        self._tone_cfgs = source_cfgs

        self._noise_cfg = noise_cfg
        self._noise_seed_v = noise_cfg["seed"] if noise_cfg else 0
        self._noise_seed_h = (noise_cfg["seed"] + 1_000_003) if noise_cfg else 0
        self._noise_std = noise_cfg["std"] if noise_cfg else 0.0

        self._current_poly: Optional[DelayPolynomial] = None
        self._delay_coeffs: Optional[np.ndarray] = None

    @property
    def channel_id_map(self) -> np.ndarray:
        return np.arange(self.num_channels)

    def tick_n_samples(self) -> int:
        """Per-channel output samples for one tick — HEAP_LEN by
        construction (BLOCK_DURATION_S is defined for exactly this)."""
        return int(round(self.channel_output_rate * BLOCK_DURATION_S))

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
        channel_output_rate) — NOT wideband input samples like
        wideband_streamer.StationStreamer.generate_next_tick takes. See
        common.ScanRunner, which sizes this via tick_n_samples() so it
        lines up with HeapAccumulator's HEAP_LEN framing exactly."""
        self._refresh_delay_poly_if_needed(t)
        poly = self._current_poly

        # Both t_local_rel_start (tone's own local clock) and
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
            out = np.zeros((n_samples, self.num_channels), dtype=np.complex128)
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

            if self._noise_cfg is not None:
                # Pure function of t, not an internal running counter — same
                # determinism/seekability property the rest of this codebase's
                # design relies on (see CLAUDE.md's sim_time discussion): any
                # pod can compute this tick's noise from t alone.
                sample_index_start = int(
                    round(t_local_rel_start * self.channel_output_rate)
                )
                out += synth_noise_all_channels(
                    noise_seed,
                    self._noise_std,
                    sample_index_start,
                    self.num_channels,
                    n_samples,
                )

            results[pol] = out

        return results


if __name__ == "__main__":
    # ============================================================
    # CORRECTNESS CHECKS
    # ============================================================

    import time

    SAMPLE_RATE_PER_CHANNEL = CHANNEL_WIDTH_HZ  # critically sampled, matches wideband_streamer

    # --- Tone: verify it lands in the expected channel, with correct
    # residual frequency and amplitude, zero delay first (sanity check
    # before testing delay).
    test_freq = (
        42 * CHANNEL_WIDTH_HZ + 150_000.0
    )  # deliberately off-center within channel 42
    zero_coeffs = np.array([0.0], dtype=np.float64)
    ch_idx, samples = synth_tone_channel(
        test_freq,
        1.0,
        BASE_FREQ_HZ,
        CHANNEL_WIDTH_HZ,
        zero_coeffs,
        0.0,
        0.0,
        False,
        0.0,
        SAMPLE_RATE_PER_CHANNEL,
        2048,
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

    # --- Tone with a KNOWN constant delay (poly = [tau_ns] only,
    # higher-order terms zero) — verify phase shift matches analytic
    # -2*pi*freq*tau exactly.
    known_tau_ns = 750.0
    coeffs = np.array([known_tau_ns], dtype=np.float64)
    ch_idx2, samples_delayed = synth_tone_channel(
        test_freq,
        1.0,
        BASE_FREQ_HZ,
        CHANNEL_WIDTH_HZ,
        coeffs,
        0.0,
        0.0,
        False,
        0.0,
        SAMPLE_RATE_PER_CHANNEL,
        2048,
    )
    expected_delayed = expected * np.exp(
        -1j * 2 * np.pi * test_freq * known_tau_ns * 1e-9
    )
    err2 = np.max(np.abs(samples_delayed - expected_delayed))
    print(f"tone known-delay phase error: {err2:.3e} (expect ~0)")
    assert err2 < 1e-9
    print("tone delay-as-phase accuracy: OK (no ring buffer needed, confirmed)")

    # --- Noise: statistics per channel, independence across channels
    noise = synth_noise_all_channels(
        seed=7, std=1.0, sample_index_start=0, num_channels=96, n_samples=200_000
    )
    print(f"noise mean: {np.mean(noise):.4f} (expect ~0)")
    print(f"noise std (real, channel 0): {np.std(noise[:, 0].real):.4f} (expect ~1.0)")
    assert abs(np.mean(noise)) < 0.01
    assert abs(np.std(noise[:, 0].real) - 1.0) < 0.01

    # cross-channel independence: correlation between two channels should be ~0
    corr = np.corrcoef(noise[:, 0].real, noise[:, 1].real)[0, 1]
    print(f"cross-channel correlation (ch0 vs ch1 real): {corr:.4f} (expect ~0)")
    assert abs(corr) < 0.02
    print("noise statistics + independence: OK")

    # determinism/seekability
    n1 = synth_noise_all_channels(7, 1.0, 1000, 96, 500)
    n2 = synth_noise_all_channels(7, 1.0, 1000, 96, 500)
    assert np.array_equal(n1, n2)
    print("noise determinism/seekability: OK")

    print("\nAll correctness checks passed.\n")

    # ============================================================
    # RELATIVE COST COMPARISON — wideband+FFT vs direct synthesis, at
    # current channel count AND projected to full 350 MHz band. This is
    # meaningful even single-threaded: we're comparing ALGORITHMIC
    # complexity (different amount of work), not parallel throughput —
    # see benchmark_direct_synthesis.py for real multi-core numbers.
    # ============================================================

    print("=" * 60)
    print("RELATIVE COST: wideband+FFT vs direct synthesis")
    print("=" * 60)

    try:
        import ska_low_station_beam_simulator.wideband_streamer as wb

        def fake_fetch(station_id, at_time):
            return DelayPolynomial(
                station_id=station_id,
                start_validity_sec=at_time,
                validity_period_sec=600.0,
                xypol_coeffs_ns=[750.0, 0.0046, 0.0, 0.0, 0.0, 0.0],
                ypol_offset_ns=2.0,
            )

        wb.fetch_delay_model_from_cbf = fake_fetch

        for num_channels_test, label in [
            (96, "current (75 MHz)"),
            (448, "full band (350 MHz)"),
        ]:
            n_samples_tick = (
                2048  # per-channel samples per tick, fixed regardless of channel count
            )
            wideband_n = n_samples_tick * num_channels_test
            sample_rate = num_channels_test * CHANNEL_WIDTH_HZ

            # --- OLD tone: wideband generation + the channelization it
            # needs to become usable (tone content isn't at channel
            # resolution until FFT'd) ---
            table = wb.build_tone_table_unit()
            tone_freqs = np.array([150_000.0], dtype=np.float64)
            tone_amps = np.array([1.0], dtype=np.float64)
            empty = np.array([], dtype=np.float64)

            wb.generate_shared_signal(
                table,
                table.shape[0],
                tone_freqs,
                tone_amps,
                empty,
                empty,
                empty,
                sample_rate,
                0.0,
                wideband_n,
            )
            t0 = time.perf_counter()
            wideband_tone = wb.generate_shared_signal(
                table,
                table.shape[0],
                tone_freqs,
                tone_amps,
                empty,
                empty,
                empty,
                sample_rate,
                0.0,
                wideband_n,
            )
            old_tone_gen_time = time.perf_counter() - t0

            ch_tone = wb.WidebandChannelizer(fft_len=num_channels_test, overlap=0)
            ch_tone.process(wideband_tone)  # warm up
            t0 = time.perf_counter()
            ch_tone.process(wideband_tone)
            old_tone_fft_time = time.perf_counter() - t0
            old_tone_total = old_tone_gen_time + old_tone_fft_time

            # --- OLD noise: wideband generation + its OWN channelization
            # (charged separately here for a fair isolated comparison,
            # even though real usage shares one FFT call across summed
            # tone+noise content) ---
            wb.generate_pol_noise(7, 1.0, 0.0, sample_rate, wideband_n)  # warm up
            t0 = time.perf_counter()
            wideband_noise = wb.generate_pol_noise(7, 1.0, 0.0, sample_rate, wideband_n)
            old_noise_gen_time = time.perf_counter() - t0

            ch_noise = wb.WidebandChannelizer(fft_len=num_channels_test, overlap=0)
            ch_noise.process(wideband_noise)  # warm up
            t0 = time.perf_counter()
            ch_noise.process(wideband_noise)
            old_noise_fft_time = time.perf_counter() - t0
            old_noise_total = old_noise_gen_time + old_noise_fft_time

            old_total = old_tone_total + old_noise_total

            # --- NEW tone: direct synthesis, O(1) regardless of channel count ---
            synth_tone_channel(
                150_000.0,
                1.0,
                BASE_FREQ_HZ,
                CHANNEL_WIDTH_HZ,
                zero_coeffs,
                0.0,
                0.0,
                False,
                0.0,
                CHANNEL_WIDTH_HZ,
                n_samples_tick,
            )
            t0 = time.perf_counter()
            synth_tone_channel(
                150_000.0,
                1.0,
                BASE_FREQ_HZ,
                CHANNEL_WIDTH_HZ,
                zero_coeffs,
                0.0,
                0.0,
                False,
                0.0,
                CHANNEL_WIDTH_HZ,
                n_samples_tick,
            )
            new_tone_time = time.perf_counter() - t0

            # --- NEW noise: direct per-channel, no FFT ever ---
            synth_noise_all_channels(7, 1.0, 0, num_channels_test, n_samples_tick)
            t0 = time.perf_counter()
            synth_noise_all_channels(7, 1.0, 0, num_channels_test, n_samples_tick)
            new_noise_time = time.perf_counter() - t0

            new_total = new_tone_time + new_noise_time

            print(f"\n{label}, {num_channels_test} channels:")
            print(
                f"  OLD tone:  gen={old_tone_gen_time * 1000:7.3f}ms  fft={old_tone_fft_time * 1000:7.3f}ms  "
                f"total={old_tone_total * 1000:7.3f}ms"
            )
            print(f"  NEW tone:  {new_tone_time * 1000:7.3f}ms")
            print(
                f"  OLD noise: gen={old_noise_gen_time * 1000:7.3f}ms  fft={old_noise_fft_time * 1000:7.3f}ms  "
                f"total={old_noise_total * 1000:7.3f}ms"
            )
            print(f"  NEW noise: {new_noise_time * 1000:7.3f}ms")
            print(
                f"  TOTAL — old={old_total * 1000:7.3f}ms  new={new_total * 1000:7.3f}ms  "
                f"ratio (old/new): {old_total / new_total:.2f}x"
            )

    except ImportError as e:
        print(f"(skipping old-vs-new comparison — could not import wideband_streamer: {e})")

    print("\nNOTE: this is 1 tone for the NEW approach vs whatever config the OLD side")
    print("uses above — not a perfectly matched workload, but the SCALING BEHAVIOR")
    print("(does cost grow with channel count or not) is the real thing to look at")
    print("here, not the absolute numbers on this machine.")
