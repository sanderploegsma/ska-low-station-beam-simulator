"""Correctness checks for ``DirectSynthesisStreamer`` and its kernels,
one assertion-group per test so a failure identifies exactly which
property broke."""

import numpy as np
import pytest

from ska_low_station_beam_simulator import direct_synthesis as sim
from ska_low_station_beam_simulator.common import (
    BASE_FREQ_HZ,
    CHANNEL_OUTPUT_RATE_HZ,
    CHANNEL_WIDTH_HZ,
    DelayFeed,
    DelayPolynomial,
    StationConfig,
)

SAMPLE_RATE_PER_CHANNEL = CHANNEL_OUTPUT_RATE_HZ  # real oversampled rate, not CHANNEL_WIDTH_HZ -- see common.py
OBS_TIME = 1_800_000_000.0


def _delay_feed(name: str, **poly_overrides) -> DelayFeed:
    """Every tone/pulsed source_cfg requires its own DelayFeed -- there is
    no default/fallback delay (see common.py). Builds one already
    populated via update(), matching how a real source would look once
    its Tango attribute subscription has delivered a first value."""
    defaults = {
        "station_id": 1,
        "start_validity_sec": OBS_TIME,
        "validity_period_sec": 600.0,
        "xypol_coeffs_ns": [750.0, 0.0046, 0.0, 0.0, 0.0, 0.0],
        "ypol_offset_ns": 2.0,
    }
    defaults.update(poly_overrides)
    feed = DelayFeed(name=name)
    feed.update(DelayPolynomial(**defaults))
    return feed


@pytest.fixture
def station():
    return StationConfig(
        station_id=1,
        substation_id=0,
        subarray_id=1,
        beam_id=1,
        first_channel_id=0,
        scan_id=1,
    )


# ============================================================
# TONE
# ============================================================


def test_tone_channel_placement():
    """A tone's energy must land in the channel bin matching its actual
    frequency -- the basic correctness check every other tone test
    depends on, since a channel-placement bug would make the
    delay/phase-accuracy checks below pass or fail for the wrong
    reason."""
    test_freq = (
        BASE_FREQ_HZ + 42 * CHANNEL_WIDTH_HZ + 150_000.0
    )  # off-center within channel 42
    zero_coeffs = np.array([0.0], dtype=np.float64)
    ch_idx, _ = sim.synth_tone_channel(
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
    assert ch_idx == 42


def test_tone_zero_delay_accuracy():
    """With zero delay applied, the synthesized tone must match the exact
    analytic complex exponential at its residual frequency -- establishes
    the undelayed baseline that test_tone_delay_as_phase_accuracy's
    delayed case below is compared against."""
    test_freq = BASE_FREQ_HZ + 42 * CHANNEL_WIDTH_HZ + 150_000.0
    zero_coeffs = np.array([0.0], dtype=np.float64)
    _, samples = sim.synth_tone_channel(
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
    residual = test_freq - (BASE_FREQ_HZ + 42 * CHANNEL_WIDTH_HZ)
    t = np.arange(2048) / SAMPLE_RATE_PER_CHANNEL
    expected = np.exp(1j * 2 * np.pi * residual * t)
    assert np.max(np.abs(samples - expected)) < 1e-9


def test_tone_delay_as_phase_accuracy():
    """Delay applied as a continuous phase term must be EXACT for a
    monochromatic tone -- not an approximation of a coarse/fine sample
    split (see direct_synthesis.py's TONE section)."""
    test_freq = BASE_FREQ_HZ + 42 * CHANNEL_WIDTH_HZ + 150_000.0
    residual = test_freq - (BASE_FREQ_HZ + 42 * CHANNEL_WIDTH_HZ)
    t = np.arange(2048) / SAMPLE_RATE_PER_CHANNEL
    expected = np.exp(1j * 2 * np.pi * residual * t)

    known_tau_ns = 750.0
    coeffs = np.array([known_tau_ns], dtype=np.float64)
    _, samples_delayed = sim.synth_tone_channel(
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
    assert np.max(np.abs(samples_delayed - expected_delayed)) < 1e-9


# ============================================================
# NOISE — fill_noise_bank (plain numpy Generator, one-time construction
# call; see module docstring's NUMBA section for why this isn't numba)
# ============================================================


def test_noise_bank_statistics_and_cross_channel_independence():
    """Confirms fill_noise_bank actually produces statistically-correct,
    mutually-uncorrelated complex Gaussian noise across channels -- the
    DFT-of-i.i.d.-Gaussian property the whole pre-generated tile-bank
    design (see direct_synthesis.py's NOISE section) relies on holding in
    practice, not just in theory."""
    bank = sim.fill_noise_bank(
        seed=7, std=1.0, n_tiles=1, tile_n_samples=200_000, num_channels=96
    )
    noise = bank[0]
    assert abs(np.mean(noise)) < 0.01
    assert abs(np.std(noise[:, 0].real) - 1.0) < 0.01
    corr = np.corrcoef(noise[:, 0].real, noise[:, 1].real)[0, 1]
    assert abs(corr) < 0.02


def test_noise_bank_determinism():
    """Generation must be a pure, reproducible function of its seed --
    required by this codebase's deterministic, clock-independent sim_time
    design (see CLAUDE.md), where any pod must be able to recompute the
    same content independently, with no shared state."""
    n1 = sim.fill_noise_bank(
        seed=7, std=1.0, n_tiles=4, tile_n_samples=500, num_channels=96
    )
    n2 = sim.fill_noise_bank(
        seed=7, std=1.0, n_tiles=4, tile_n_samples=500, num_channels=96
    )
    assert np.array_equal(n1, n2)


def test_noise_bank_different_seeds_differ():
    """Complements the determinism check above: different seeds must
    actually produce different noise, guarding against a degenerate
    implementation that ignores the seed and always returns the same
    bank content."""
    n1 = sim.fill_noise_bank(
        seed=7, std=1.0, n_tiles=4, tile_n_samples=500, num_channels=96
    )
    n2 = sim.fill_noise_bank(
        seed=99, std=1.0, n_tiles=4, tile_n_samples=500, num_channels=96
    )
    assert not np.array_equal(n1, n2)


# ============================================================
# NOISE TILE BANK (via the full streamer)
# ============================================================


def test_tile_bank_determinism(station):
    """Same determinism requirement as the raw-kernel checks above, but
    exercised through the full DirectSynthesisStreamer/tile-bank path:
    re-requesting the same tick time must return byte-identical
    content."""
    streamer = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[],
        noise_cfg={"std": 1.0, "seed": 7},
        obs_time_ref=OBS_TIME,
        num_channels=96,
        n_tiles=8,
    )
    n = streamer.tick_n_samples()
    r1 = streamer.generate_next_tick(OBS_TIME, n)["V"].copy()
    r2 = streamer.generate_next_tick(OBS_TIME, n)["V"].copy()
    assert np.array_equal(r1, r2), "same t must give identical content"


def test_tile_bank_first_repeat_is_early_for_small_n_tiles(station):
    """Not a fidelity bug -- the birthday-paradox tradeoff this bank
    deliberately makes (see direct_synthesis.py's NOISE section)."""
    streamer = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[],
        noise_cfg={"std": 1.0, "seed": 7},
        obs_time_ref=OBS_TIME,
        num_channels=96,
        n_tiles=8,
    )
    n = streamer.tick_n_samples()
    tick_dt = n / streamer.channel_output_rate
    seen = {}
    first_repeat_at = None
    for i in range(200):
        out = streamer.generate_next_tick(OBS_TIME + i * tick_dt, n)["V"]
        key = out[0, 0]
        if key in seen and first_repeat_at is None:
            first_repeat_at = i
        seen[key] = i
    assert first_repeat_at is not None
    assert first_repeat_at < 100


def test_tile_bank_cross_station_independence(station):
    """Non-negotiable per the module docstring: stations must NEVER emit
    byte-identical noise for the same tick."""
    streamer_a = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[],
        noise_cfg={"std": 1.0, "seed": 7},
        obs_time_ref=OBS_TIME,
        num_channels=96,
        n_tiles=8,
    )
    streamer_b = sim.DirectSynthesisStreamer(
        station=StationConfig(
            station_id=2,
            substation_id=0,
            subarray_id=1,
            beam_id=1,
            first_channel_id=0,
            scan_id=1,
        ),
        source_cfgs=[],
        noise_cfg={"std": 1.0, "seed": 99},
        obs_time_ref=OBS_TIME,
        num_channels=96,
        n_tiles=8,
    )
    n = streamer_a.tick_n_samples()
    out_a = streamer_a.generate_next_tick(OBS_TIME, n)["V"]
    out_b = streamer_b.generate_next_tick(OBS_TIME, n)["V"]
    assert not np.array_equal(out_a, out_b)


# ============================================================
# PULSAR: channel-mapping ground truth
# ============================================================


def test_pulsar_channelize_once_channel_mapping():
    """Regression test for bug #14 (rfft/irfft convention bug): inject a
    KNOWN tone at a known external channel and confirm it lands there,
    rather than trusting the FFT-bin relabeling logic by inspection."""
    num_channels = 32
    chw = CHANNEL_WIDTH_HZ
    base_f = sim.BASE_FREQ_HZ
    wideband_rate = num_channels * chw
    n_wide = 256 * num_channels
    test_channel = 7
    band_center_hz = base_f + wideband_rate / 2.0
    test_freq = (
        base_f + test_channel * chw + 0.15 * chw
    )  # off-center within the channel
    f_offset = test_freq - band_center_hz
    t_wide = np.arange(n_wide) / wideband_rate
    v_tone = np.exp(1j * 2 * np.pi * f_offset * t_wide)
    channelized = sim._channelize_once(v_tone, num_channels)
    power_per_channel = np.mean(np.abs(channelized) ** 2, axis=0)
    detected_channel = int(np.argmax(power_per_channel))
    assert detected_channel == test_channel


# ============================================================
# PULSAR: dispersion constant cross-check
# ============================================================


def test_dispersion_constant_matches_psrsigsim():
    """Cross-validated against an external, peer-reviewed reference, not
    just internal self-consistency -- see direct_synthesis.py's PULSED
    section for why PsrSigSim itself wasn't taken as a dependency."""
    psrsigsim_dm_k = 1.0 / 2.41e-4
    rel_diff = abs(sim.DISPERSION_CONST_S_MHZ2_PER_DM - psrsigsim_dm_k) / psrsigsim_dm_k
    assert (
        rel_diff < 0.001
    )  # ~0.014% in practice -- standard literature-precision variation


# ============================================================
# PULSAR: full pipeline + coherence
# ============================================================

PULSAR_NUM_CHANNELS = 384  # the real ICD maximum -- was 448 (an invalid config)
PULSAR_DM = 2.0
PULSAR_PERIOD_S = 0.1
PULSAR_WIDTH_S = 0.005


@pytest.fixture
def pulsar_streamer(station):
    return sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {
                "kind": "pulsed",
                "period_s": PULSAR_PERIOD_S,
                "width_s": PULSAR_WIDTH_S,
                "amplitude": 1.0,
                "dm_pc_cm3": PULSAR_DM,
                "delay_feed": _delay_feed("pulsar-a"),
            }
        ],
        obs_time_ref=OBS_TIME,
        num_channels=PULSAR_NUM_CHANNELS,
        base_freq_hz=sim.BASE_FREQ_HZ,
    )


def test_pulsar_determinism(pulsar_streamer):
    """Same determinism requirement as the noise tile bank's, applied to
    the pulsar path -- generate_next_tick must be a pure function of tick
    time so any station pod can recompute a tick independently without
    shared state or a wall clock."""
    n = pulsar_streamer.tick_n_samples()
    rp1 = pulsar_streamer.generate_next_tick(OBS_TIME, n)["V"].copy()
    rp2 = pulsar_streamer.generate_next_tick(OBS_TIME, n)["V"].copy()
    assert np.array_equal(rp1, rp2)


def test_pulsar_content_is_genuinely_complex(pulsar_streamer):
    """v2's real-valued-only content couldn't be coherently beamformed --
    this is the property that fixed it (see PULSED section, v3)."""
    n = pulsar_streamer.tick_n_samples()
    out = pulsar_streamer.generate_next_tick(OBS_TIME, n)["V"]
    assert np.max(np.abs(out.imag)) > 1e-6


def test_pulsar_cross_station_coherence_after_delay_compensation(
    pulsar_streamer, station
):
    """The concrete confirmation that v3 supports coherent multi-station
    beamforming: two streamers with different DelayPolynomials, same
    pulsar, same tick -- after each applies its OWN delay-compensating
    phase correction (what a beamformer does), content must correlate
    at ~1.0."""
    n = pulsar_streamer.tick_n_samples()
    station_b = StationConfig(
        station_id=2,
        substation_id=0,
        subarray_id=1,
        beam_id=1,
        first_channel_id=0,
        scan_id=1,
    )

    pulsar_streamer_b = sim.DirectSynthesisStreamer(
        station=station_b,
        source_cfgs=[
            {
                "kind": "pulsed",
                "period_s": PULSAR_PERIOD_S,
                "width_s": PULSAR_WIDTH_S,
                "amplitude": 1.0,
                "dm_pc_cm3": PULSAR_DM,
                "delay_feed": _delay_feed(
                    "pulsar-b",
                    station_id=2,
                    xypol_coeffs_ns=[300.0, 0.002, 0.0, 0.0, 0.0, 0.0],
                    ypol_offset_ns=1.0,
                ),
            }
        ],
        obs_time_ref=OBS_TIME,
        num_channels=PULSAR_NUM_CHANNELS,
        base_freq_hz=sim.BASE_FREQ_HZ,
    )

    out_b = pulsar_streamer_b.generate_next_tick(OBS_TIME, n)["V"]
    out_a = pulsar_streamer.generate_next_tick(OBS_TIME, n)["V"]

    test_ch = 200
    poly_a = pulsar_streamer._pulsars[0][3].get(OBS_TIME)
    poly_b = pulsar_streamer_b._pulsars[0][3].get(OBS_TIME)
    tau_a_ns = sim.eval_delay_poly_ns(
        np.asarray(poly_a.xypol_coeffs_ns), OBS_TIME - poly_a.start_validity_sec
    )
    tau_b_ns = sim.eval_delay_poly_ns(
        np.asarray(poly_b.xypol_coeffs_ns), OBS_TIME - poly_b.start_validity_sec
    )
    f_c = pulsar_streamer.base_freq_hz + test_ch * pulsar_streamer.channel_width_hz
    correction_a = np.exp(1j * 2 * np.pi * f_c * tau_a_ns * 1e-9)
    correction_b = np.exp(1j * 2 * np.pi * f_c * tau_b_ns * 1e-9)
    aligned_a = out_a[:, test_ch] * correction_a
    aligned_b = out_b[:, test_ch] * correction_b
    coh = np.abs(np.vdot(aligned_a, aligned_b)) / np.sqrt(
        np.vdot(aligned_a, aligned_a).real * np.vdot(aligned_b, aligned_b).real
    )
    assert coh > 0.999


# ============================================================
# num_channels validation (SPS-CBF ICD: 8-384 in steps of 8)
# ============================================================


def test_num_channels_above_max_rejected(station):
    """The SPS-CBF ICD caps a beam at 384 channels -- a request above
    that must be rejected at construction time rather than silently
    producing an invalid beam configuration."""
    with pytest.raises(ValueError, match="not a valid SPS beam"):
        sim.DirectSynthesisStreamer(
            station=station,
            source_cfgs=[],
            obs_time_ref=OBS_TIME,
            num_channels=448,
        )


def test_num_channels_not_a_multiple_of_step_rejected(station):
    """The ICD also requires num_channels to land on an 8-channel step --
    an off-grid value must be rejected outright rather than silently
    rounded or accepted."""
    with pytest.raises(ValueError, match="not a valid SPS beam"):
        sim.DirectSynthesisStreamer(
            station=station,
            source_cfgs=[],
            obs_time_ref=OBS_TIME,
            num_channels=100,
        )


def test_num_channels_at_max_accepted(station):
    """Confirms the ICD's actual maximum (384) is itself a valid,
    constructible configuration, not just that values above it are
    rejected -- guards against an off-by-one in the validation bound."""
    sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[],
        obs_time_ref=OBS_TIME,
        num_channels=384,
    )


# ============================================================
# PULSAR: loading a pre-generated catalog entry by name
# ============================================================


CATALOG_TEST_NUM_CHANNELS = 32
CATALOG_TEST_PERIOD_S = 0.005
CATALOG_TEST_WIDTH_S = 0.0005
CATALOG_TEST_DM = 3.0


@pytest.fixture(scope="module")
def small_catalog(tmp_path_factory):
    """A real (not synthetic) catalog entry, but at a small num_channels
    and short period so it builds fast -- module-scoped so the (still
    non-trivial) build cost is paid once for every test that needs it,
    not once per test."""
    from ska_low_station_beam_simulator.pulsar_catalog import save_pulsar_to_catalog

    catalog_dir = tmp_path_factory.mktemp("pulsar_catalog")
    template, n_period_samples = sim.build_pulsar_template(
        CATALOG_TEST_NUM_CHANNELS,
        CHANNEL_WIDTH_HZ,
        BASE_FREQ_HZ,
        CHANNEL_OUTPUT_RATE_HZ,
        CATALOG_TEST_PERIOD_S,
        CATALOG_TEST_WIDTH_S,
        1.0,
        CATALOG_TEST_DM,
        sim.DEFAULT_SKY_SEED,
    )
    save_pulsar_to_catalog(
        catalog_dir,
        "test_catalog_pulsar",
        template,
        n_period_samples,
        CATALOG_TEST_PERIOD_S,
        CATALOG_TEST_WIDTH_S,
        CATALOG_TEST_DM,
        sim.DEFAULT_SKY_SEED,
        num_channels=CATALOG_TEST_NUM_CHANNELS,
        base_freq_hz=BASE_FREQ_HZ,
    )
    return catalog_dir, template, n_period_samples


def test_pulsed_source_cfg_rejects_both_name_and_params(station):
    """A pulsed source_cfg naming a catalog entry AND supplying direct
    period_s/width_s/dm_pc_cm3 params is ambiguous about which should
    win -- must be rejected rather than silently preferring one over the
    other."""
    with pytest.raises(ValueError, match="both"):
        sim.DirectSynthesisStreamer(
            station=station,
            source_cfgs=[
                {
                    "kind": "pulsed",
                    "pulsar_name": "x",
                    "period_s": 0.01,
                    "width_s": 0.001,
                    "dm_pc_cm3": 2.0,
                    "delay_feed": _delay_feed("both"),
                }
            ],
            obs_time_ref=OBS_TIME,
        )


def test_pulsed_source_cfg_rejects_neither_name_nor_params(station):
    """A pulsed source_cfg giving neither a catalog name nor direct params
    has no way to know what pulsar to build -- must fail loudly at
    construction rather than produce undefined content."""
    with pytest.raises(ValueError, match="neither"):
        sim.DirectSynthesisStreamer(
            station=station,
            source_cfgs=[{"kind": "pulsed", "delay_feed": _delay_feed("neither")}],
            obs_time_ref=OBS_TIME,
        )


def test_pulsar_name_loads_and_generates_ticks(small_catalog, station):
    """Basic end-to-end check that loading a pulsar by catalog name
    produces a working streamer: deterministic per-tick content, and
    genuinely complex output -- v2's real-valued-only content couldn't be
    coherently beamformed across stations (see the PULSED section), so
    this is checked here too, not just on a directly-built template."""
    catalog_dir, _, _ = small_catalog
    streamer = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {
                "kind": "pulsed",
                "pulsar_name": "test_catalog_pulsar",
                "catalog_dir": catalog_dir,
                "delay_feed": _delay_feed("catalog-pulsar"),
            }
        ],
        obs_time_ref=OBS_TIME,
        num_channels=CATALOG_TEST_NUM_CHANNELS,
        base_freq_hz=BASE_FREQ_HZ,
    )
    n = streamer.tick_n_samples()
    out1 = streamer.generate_next_tick(OBS_TIME, n)["V"].copy()
    out2 = streamer.generate_next_tick(OBS_TIME, n)["V"].copy()
    assert np.array_equal(out1, out2), "same t must give identical content"
    assert np.max(np.abs(out1.imag)) > 1e-6, (
        "loaded template must still be genuinely complex"
    )


def test_pulsar_name_matches_directly_built_content(small_catalog, station):
    """The by-name path must produce the SAME content a direct
    period_s/width_s/dm_pc_cm3 build would (modulo complex64 round-trip
    precision) -- this is the property that makes 'fast startup via a
    catalog' a packaging optimization, not a different physical result."""
    catalog_dir, _, _ = small_catalog

    streamer_by_name = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {
                "kind": "pulsed",
                "pulsar_name": "test_catalog_pulsar",
                "catalog_dir": catalog_dir,
                "delay_feed": _delay_feed("by-name"),
            }
        ],
        obs_time_ref=OBS_TIME,
        num_channels=CATALOG_TEST_NUM_CHANNELS,
        base_freq_hz=BASE_FREQ_HZ,
    )
    streamer_direct = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {
                "kind": "pulsed",
                "period_s": CATALOG_TEST_PERIOD_S,
                "width_s": CATALOG_TEST_WIDTH_S,
                "dm_pc_cm3": CATALOG_TEST_DM,
                "delay_feed": _delay_feed("direct"),
            }
        ],
        obs_time_ref=OBS_TIME,
        num_channels=CATALOG_TEST_NUM_CHANNELS,
        base_freq_hz=BASE_FREQ_HZ,
    )

    n = streamer_by_name.tick_n_samples()
    out_by_name = streamer_by_name.generate_next_tick(OBS_TIME, n)["V"]
    out_direct = streamer_direct.generate_next_tick(OBS_TIME, n)["V"]
    assert np.allclose(out_by_name, out_direct, atol=1e-3)


def test_pulsar_name_amplitude_override_scales_content(small_catalog, station):
    """The by-name catalog path still needs to support scaling a stored
    template's amplitude at load time -- confirms the override is applied
    as a linear scale factor, rather than being silently ignored once a
    template comes from disk instead of a direct build."""
    catalog_dir, _, _ = small_catalog
    streamer_default = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {
                "kind": "pulsed",
                "pulsar_name": "test_catalog_pulsar",
                "catalog_dir": catalog_dir,
                "delay_feed": _delay_feed("amp-default"),
            }
        ],
        obs_time_ref=OBS_TIME,
        num_channels=CATALOG_TEST_NUM_CHANNELS,
        base_freq_hz=BASE_FREQ_HZ,
    )
    streamer_scaled = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {
                "kind": "pulsed",
                "pulsar_name": "test_catalog_pulsar",
                "catalog_dir": catalog_dir,
                "amplitude": 2.5,
                "delay_feed": _delay_feed("amp-scaled"),
            }
        ],
        obs_time_ref=OBS_TIME,
        num_channels=CATALOG_TEST_NUM_CHANNELS,
        base_freq_hz=BASE_FREQ_HZ,
    )
    n = streamer_default.tick_n_samples()
    out_default = streamer_default.generate_next_tick(OBS_TIME, n)["V"]
    out_scaled = streamer_scaled.generate_next_tick(OBS_TIME, n)["V"]
    assert np.allclose(out_scaled, out_default * 2.5, atol=1e-3)
