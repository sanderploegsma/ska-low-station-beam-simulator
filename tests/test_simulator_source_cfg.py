"""Tests for ``simulator.build_source_cfg`` -- the JSON-boundary dispatch
that turns one raw ``source_cfgs`` entry from ``StartScan``'s JSON
argument into a typed ``direct_synthesis`` config
(``ToneSourceConfig``/``PulsarByNameConfig``/``PulsarByParamsConfig``).

Kept as a standalone, Tango-free function specifically so this
JSON-shape validation (unsupported ``kind``, ambiguous/incomplete pulsed
source) is unit-testable without standing up a real Tango device -- see
``simulator.build_source_cfg``'s own docstring and CLAUDE.md's Setup
section (this codebase otherwise doesn't unit test the Tango device
server layer at all).
"""

import pytest

from ska_low_station_beam_simulator import direct_synthesis as ds
from ska_low_station_beam_simulator import simulator as sim
from ska_low_station_beam_simulator.common import DelayFeed


def _feed(name: str = "test-feed") -> DelayFeed:
    return DelayFeed(name=name)


def test_build_source_cfg_tone():
    feed = _feed()
    cfg = sim.build_source_cfg(
        {"kind": "tone", "freq_hz": 60e6, "amplitude": 2.0}, feed
    )
    assert isinstance(cfg, ds.ToneSourceConfig)
    assert cfg.freq_hz == 60e6
    assert cfg.amplitude == 2.0
    assert cfg.delay_feed is feed


def test_build_source_cfg_pulsed_by_name():
    feed = _feed()
    cfg = sim.build_source_cfg(
        {"kind": "pulsed", "pulsar_name": "j0000+0000"}, feed
    )
    assert isinstance(cfg, ds.PulsarByNameConfig)
    assert cfg.pulsar_name == "j0000+0000"
    assert cfg.delay_feed is feed


def test_build_source_cfg_pulsed_by_params():
    feed = _feed()
    cfg = sim.build_source_cfg(
        {
            "kind": "pulsed",
            "period_s": 0.1,
            "width_s": 0.005,
            "dm_pc_cm3": 2.0,
        },
        feed,
    )
    assert isinstance(cfg, ds.PulsarByParamsConfig)
    assert cfg.period_s == 0.1
    assert cfg.width_s == 0.005
    assert cfg.dm_pc_cm3 == 2.0
    assert cfg.delay_feed is feed


def test_build_source_cfg_ignores_delay_attr_uri():
    """``delay_attr_uri`` is consumed by StartScan to build the DelayFeed
    before calling build_source_cfg -- it must not be forwarded as a
    dataclass field."""
    feed = _feed()
    cfg = sim.build_source_cfg(
        {"kind": "tone", "freq_hz": 60e6, "delay_attr_uri": "some/attr"}, feed
    )
    assert isinstance(cfg, ds.ToneSourceConfig)


def test_build_source_cfg_rejects_unsupported_kind():
    with pytest.raises(ValueError, match="not supported"):
        sim.build_source_cfg({"kind": "wideband"}, _feed())


def test_build_source_cfg_rejects_pulsed_with_both_name_and_params():
    """A pulsed source_cfg naming a catalog entry AND supplying direct
    period_s/width_s/dm_pc_cm3 params is ambiguous about which should
    win -- must be rejected rather than silently preferring one over the
    other."""
    with pytest.raises(ValueError, match="both"):
        sim.build_source_cfg(
            {
                "kind": "pulsed",
                "pulsar_name": "x",
                "period_s": 0.01,
                "width_s": 0.001,
                "dm_pc_cm3": 2.0,
            },
            _feed(),
        )


def test_build_source_cfg_rejects_pulsed_with_neither_name_nor_params():
    """A pulsed source_cfg giving neither a catalog name nor direct params
    has no way to know what pulsar to build -- must fail loudly rather
    than produce undefined content."""
    with pytest.raises(ValueError, match="neither"):
        sim.build_source_cfg({"kind": "pulsed"}, _feed())
