"""Tests for pulsar_catalog.py's save/load/slice logic, using small
synthetic templates rather than real (slow, hundreds-of-MB) pulsar
templates from build_pulsar_template -- these tests are about the
catalog's own bookkeeping (slicing math, stale-constant detection,
error messages), not pulsar physics, which tests/test_direct_synthesis.py
already covers.
"""

import numpy as np
import pytest

from ska_low_station_beam_simulator.common import (
    BASE_FREQ_HZ,
    CHANNEL_OUTPUT_RATE_HZ,
    CHANNEL_WIDTH_HZ,
)
from ska_low_station_beam_simulator.pulsar_catalog import (
    load_pulsar_from_catalog,
    save_pulsar_to_catalog,
)

CATALOG_NUM_CHANNELS = 384
N_PERIOD_SAMPLES = 100


def _make_synthetic_template() -> np.ndarray:
    """Each channel's content is a distinct constant (its own channel
    index) -- makes slicing bugs (off-by-one, wrong axis) immediately
    obvious rather than needing a numerically-close comparison."""
    return np.broadcast_to(
        np.arange(CATALOG_NUM_CHANNELS, dtype=np.complex128)[:, None],
        (CATALOG_NUM_CHANNELS, N_PERIOD_SAMPLES),
    ).copy()


def _save(tmp_path, name="test_pulsar", **overrides):
    template = overrides.pop("template", _make_synthetic_template())
    kwargs = {
        "n_period_samples": N_PERIOD_SAMPLES,
        "period_s": 0.05,
        "width_s": 0.002,
        "dm_pc_cm3": 10.0,
        "sky_seed": 1234,
        "num_channels": CATALOG_NUM_CHANNELS,
        "base_freq_hz": BASE_FREQ_HZ,
        "channel_width_hz": CHANNEL_WIDTH_HZ,
        "channel_output_rate": CHANNEL_OUTPUT_RATE_HZ,
    }
    kwargs.update(overrides)
    save_pulsar_to_catalog(tmp_path, name, template, **kwargs)
    return template


def test_load_full_range_matches_saved_template(tmp_path):
    """Baseline round-trip check: loading a catalog entry across its full
    channel range must reproduce exactly what was saved (modulo the
    expected complex64-storage precision) -- every other test in this
    module exercises narrower slices or failure modes built on top of
    this working correctly."""
    template = _save(tmp_path)
    loaded = load_pulsar_from_catalog(
        "test_pulsar", CATALOG_NUM_CHANNELS, BASE_FREQ_HZ, catalog_dir=tmp_path
    )
    # complex64 round-trip loses precision -- values here are small
    # integers, well within complex64's exact-representation range, so
    # this can and should be an exact match.
    assert np.array_equal(loaded.template, template)
    assert loaded.template.dtype == np.complex128
    assert loaded.period_s == 0.05
    assert loaded.n_period_samples == N_PERIOD_SAMPLES
    assert loaded.dm_pc_cm3 == 10.0


def test_load_slices_correct_channel_range(tmp_path):
    """Loading a sub-band from a wider catalog entry must slice the
    correct CHANNELS, not just the correct count of them -- uses the
    synthetic template's per-channel-is-its-own-index content so a
    wrong-offset slice is immediately obvious rather than needing a
    numerically-close comparison."""
    _save(tmp_path)
    slice_start = 50
    n_channels = 32
    loaded = load_pulsar_from_catalog(
        "test_pulsar",
        n_channels,
        BASE_FREQ_HZ + slice_start * CHANNEL_WIDTH_HZ,
        catalog_dir=tmp_path,
    )
    template = loaded.template
    assert template.shape == (n_channels, N_PERIOD_SAMPLES)
    # channel c's content is literally its GLOBAL channel index -- so a
    # slice starting at channel 50 should read 50, 51, 52, ...
    assert np.array_equal(
        template[:, 0], np.arange(slice_start, slice_start + n_channels)
    )


def test_load_rejects_misaligned_base_freq(tmp_path):
    """A requested base_freq_hz that doesn't land on a channel boundary
    can't be sliced unambiguously -- must be rejected rather than
    silently rounded to some nearby channel."""
    _save(tmp_path)
    with pytest.raises(ValueError, match="not aligned"):
        load_pulsar_from_catalog(
            "test_pulsar",
            32,
            BASE_FREQ_HZ + 0.3 * CHANNEL_WIDTH_HZ,
            catalog_dir=tmp_path,
        )


def test_load_rejects_out_of_range_slice(tmp_path):
    """A requested channel range extending past the end of the saved
    catalog entry has no data to serve -- must fail loudly rather than
    silently returning a short or wrapped-around slice."""
    _save(tmp_path)
    with pytest.raises(ValueError, match="doesn't fit"):
        load_pulsar_from_catalog(
            "test_pulsar",
            CATALOG_NUM_CHANNELS,
            BASE_FREQ_HZ
            + 10 * CHANNEL_WIDTH_HZ,  # pushes the end past the catalog's range
            catalog_dir=tmp_path,
        )


def test_load_rejects_below_catalog_band(tmp_path):
    """Same out-of-range guard as test_load_rejects_out_of_range_slice
    above, for a request starting below the catalog entry's stored band
    -- both directions of out-of-range must be caught, not just one."""
    _save(tmp_path)
    with pytest.raises(ValueError, match="doesn't fit"):
        load_pulsar_from_catalog(
            "test_pulsar",
            32,
            BASE_FREQ_HZ - 10 * CHANNEL_WIDTH_HZ,
            catalog_dir=tmp_path,
        )


def test_load_rejects_stale_channel_output_rate(tmp_path):
    """Regression guard for exactly the kind of drift that silently broke
    this project's own timing once already (see CLAUDE.md's 'SPS-CBF ICD
    channelization' section) -- a catalog baked under an old/wrong
    channel_output_rate must fail loudly, not silently misapply."""
    _save(tmp_path, channel_output_rate=CHANNEL_OUTPUT_RATE_HZ * 0.9)
    with pytest.raises(ValueError, match="channel_output_rate"):
        load_pulsar_from_catalog(
            "test_pulsar", CATALOG_NUM_CHANNELS, BASE_FREQ_HZ, catalog_dir=tmp_path
        )


def test_load_rejects_stale_channel_width(tmp_path):
    """A catalog entry baked under a different channel_width_hz than the
    caller now uses would misalign every channel's frequency if loaded
    anyway -- must be rejected, the same class of drift
    test_load_rejects_stale_channel_output_rate guards against for the
    sample rate."""
    _save(tmp_path, channel_width_hz=CHANNEL_WIDTH_HZ * 2)
    with pytest.raises(ValueError, match="channel_width_hz"):
        load_pulsar_from_catalog(
            "test_pulsar", CATALOG_NUM_CHANNELS, BASE_FREQ_HZ, catalog_dir=tmp_path
        )


def test_load_rejects_unknown_name(tmp_path):
    """Requesting a pulsar name that was never saved to this catalog must
    fail with a clear, specific error, not an obscure KeyError or a
    silently-empty result."""
    _save(tmp_path)
    with pytest.raises(ValueError, match="not in catalog"):
        load_pulsar_from_catalog("nonexistent", 32, BASE_FREQ_HZ, catalog_dir=tmp_path)


def test_load_rejects_missing_catalog(tmp_path):
    """Loading from a directory with no catalog.json at all (e.g. a
    fresh or misconfigured deployment) must fail with a clear, specific
    message rather than a generic file-not-found traceback."""
    with pytest.raises(ValueError, match="no pulsar catalog found"):
        load_pulsar_from_catalog("anything", 32, BASE_FREQ_HZ, catalog_dir=tmp_path)


def test_save_preserves_other_entries(tmp_path):
    """generate_pulsar_catalog.py calls save_pulsar_to_catalog once per
    entry -- a later save must not clobber earlier entries in the shared
    catalog.json."""
    _save(tmp_path, name="pulsar_a")
    _save(tmp_path, name="pulsar_b")
    for name in ("pulsar_a", "pulsar_b"):
        loaded = load_pulsar_from_catalog(name, 32, BASE_FREQ_HZ, catalog_dir=tmp_path)
        assert loaded.period_s == 0.05
