"""Tests for ``simulator.build_tone_source_request`` -- the JSON-boundary
dispatch that turns one raw ``source_cfgs`` entry from ``StartScan``'s
JSON argument into a ``ToneSourceConfig`` protobuf message for the Go
gRPC simulator.

Kept as a standalone, Tango-free (and gRPC-free) function specifically
so this JSON-shape validation (unsupported ``kind``, missing
``delay_attr_uri``) is unit-testable without standing up a live Tango
device or a live gRPC server -- see
``simulator.build_tone_source_request``'s own docstring and CLAUDE.md's
Setup section (this codebase otherwise doesn't unit test the Tango
device server layer at all).
"""

import pytest

from ska_low_station_beam_simulator import simulator as sim


def test_build_tone_source_request():
    req = sim.build_tone_source_request(
        {
            "kind": "tone",
            "freq_hz": 60e6,
            "amplitude": 2.0,
            "delay_attr_uri": "sys/delaypoly/1/direction0",
        }
    )
    assert req.freq_hz == 60e6
    assert req.amplitude == 2.0
    # source_id reuses delay_attr_uri verbatim -- see module docstring
    # for why (already required, already unique per source).
    assert req.source_id == "sys/delaypoly/1/direction0"


def test_build_tone_source_request_omits_delay_attr_uri_as_a_stray_field():
    """``delay_attr_uri`` becomes ``source_id`` -- it must not also be
    forwarded as some other protobuf field (there is none named that, so
    this would raise if it leaked through)."""
    req = sim.build_tone_source_request(
        {"kind": "tone", "freq_hz": 60e6, "delay_attr_uri": "some/attr"}
    )
    assert req.source_id == "some/attr"


def test_build_tone_source_request_rejects_pulsed():
    """The Go gRPC backend doesn't support pulsar sources yet (see
    api/simulator.proto's doc comment) -- a 'pulsed' entry must fail
    loudly, not be silently dropped or misrouted as a tone."""
    with pytest.raises(ValueError, match="not supported"):
        sim.build_tone_source_request({"kind": "pulsed", "pulsar_name": "j0000+0000"})


def test_build_tone_source_request_rejects_unsupported_kind():
    with pytest.raises(ValueError, match="not supported"):
        sim.build_tone_source_request({"kind": "wideband"})


def test_build_tone_source_request_rejects_missing_delay_attr_uri():
    with pytest.raises(ValueError, match="delay_attr_uri"):
        sim.build_tone_source_request({"kind": "tone", "freq_hz": 60e6})
