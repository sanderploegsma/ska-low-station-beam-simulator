"""Tests for common.py's per-source delay feed abstraction (DelayFeed,
parse_delay_polynomial_from_attr_value) and its wiring into
DirectSynthesisStreamer.

These back the per-source delay-polynomial feature: every tone/pulsar now
gets its own DelayFeed instead of sharing one station-level polynomial,
which is what lets two sources at different (simulated) sky directions
receive genuinely different delay corrections in the same streamer.
There is deliberately no default/fallback feed — a source with no real
delay path would silently produce content that's trivially "perfectly
aligned", which could mask a real CBF delay-tracking bug rather than
exercise it — so DirectSynthesisStreamer requires one explicitly.
"""

import json
import logging

import numpy as np

from ska_low_station_beam_simulator import direct_synthesis as sim
from ska_low_station_beam_simulator.common import (
    CHANNEL_WIDTH_HZ,
    DelayFeed,
    DelayPolynomial,
    StationConfig,
    parse_delay_polynomial_from_attr_value,
)


def _poly(**overrides):
    defaults = dict(
        station_id=1,
        start_validity_sec=0.0,
        validity_period_sec=600.0,
        xypol_coeffs_ns=[100.0],
        ypol_offset_ns=0.0,
    )
    defaults.update(overrides)
    return DelayPolynomial(**defaults)


# ============================================================
# DelayFeed
# ============================================================


def test_delay_feed_defaults_to_zero_delay_before_first_update(caplog):
    feed = DelayFeed(name="test-source")
    with caplog.at_level(logging.WARNING, logger="cbf_sim"):
        poly = feed.get(1000.0)
    assert sum(poly.xypol_coeffs_ns) == 0.0
    assert poly.ypol_offset_ns == 0.0
    assert any("has not received a polynomial yet" in r.message for r in caplog.records)


def test_delay_feed_warns_only_once_for_missing_poly(caplog):
    feed = DelayFeed(name="test-source")
    with caplog.at_level(logging.WARNING, logger="cbf_sim"):
        feed.get(1000.0)
        feed.get(1001.0)
        feed.get(1002.0)
    warnings = [r for r in caplog.records if "has not received a polynomial yet" in r.message]
    assert len(warnings) == 1


def test_delay_feed_applies_updated_polynomial():
    feed = DelayFeed(name="test-source")
    feed.update(_poly(xypol_coeffs_ns=[750.0]))
    assert feed.get(0.0).xypol_coeffs_ns == [750.0]


def test_delay_feed_keeps_applying_expired_polynomial_with_warning(caplog):
    """Design decision: a stalled upstream publisher is not this
    simulator's problem to fix -- it applies whatever it was last told
    and logs that the delay is known to be stale."""
    feed = DelayFeed(name="test-source")
    feed.update(_poly(start_validity_sec=0.0, validity_period_sec=10.0, xypol_coeffs_ns=[750.0]))
    with caplog.at_level(logging.WARNING, logger="cbf_sim"):
        poly = feed.get(20.0)  # past valid_until=10.0
    assert poly.xypol_coeffs_ns == [750.0]
    assert any("expired" in r.message for r in caplog.records)


def test_delay_feed_warns_once_per_staleness_episode(caplog):
    feed = DelayFeed(name="test-source")
    feed.update(_poly(start_validity_sec=0.0, validity_period_sec=10.0))
    with caplog.at_level(logging.WARNING, logger="cbf_sim"):
        feed.get(20.0)
        feed.get(21.0)
        feed.get(22.0)
    stale_warnings = [r for r in caplog.records if "expired" in r.message]
    assert len(stale_warnings) == 1

    # A fresh update clears the staleness flag, so a later expiry of the
    # NEW polynomial must warn again, independently.
    feed.update(_poly(start_validity_sec=20.0, validity_period_sec=5.0))
    with caplog.at_level(logging.WARNING, logger="cbf_sim"):
        feed.get(30.0)
    stale_warnings = [r for r in caplog.records if "expired" in r.message]
    assert len(stale_warnings) == 2


# ============================================================
# DirectSynthesisStreamer requires a delay_feed per source
# ============================================================


def test_streamer_rejects_tone_without_delay_feed():
    station = StationConfig(
        station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1
    )
    try:
        sim.DirectSynthesisStreamer(
            station=station,
            source_cfgs=[{"kind": "tone", "freq_hz": 60e6, "amplitude": 1.0}],
            obs_time_ref=0.0,
            num_channels=32,
        )
    except ValueError as exc:
        assert "delay_feed" in str(exc)
    else:
        raise AssertionError("expected ValueError for a source_cfg missing delay_feed")


def test_streamer_rejects_pulsed_without_delay_feed():
    station = StationConfig(
        station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1
    )
    try:
        sim.DirectSynthesisStreamer(
            station=station,
            source_cfgs=[
                {"kind": "pulsed", "period_s": 0.1, "width_s": 0.005, "amplitude": 1.0, "dm_pc_cm3": 2.0}
            ],
            obs_time_ref=0.0,
            num_channels=32,
            base_freq_hz=sim.BASE_FREQ_HZ,
        )
    except ValueError as exc:
        assert "delay_feed" in str(exc)
    else:
        raise AssertionError("expected ValueError for a source_cfg missing delay_feed")


# ============================================================
# parse_delay_polynomial_from_attr_value — UNVERIFIED wire format (see
# its docstring); this only pins down the assumed JSON shape, not the
# real ska-low-csp-delaymodel/1.0 schema.
# ============================================================


def test_parse_delay_polynomial_from_json_string():
    payload = json.dumps(
        {
            "start_validity_sec": 123.0,
            "validity_period_sec": 600.0,
            "xypol_coeffs_ns": [1.0, 2.0, 3.0],
            "ypol_offset_ns": 4.0,
        }
    )
    poly = parse_delay_polynomial_from_attr_value(payload, station_id=5)
    assert poly.station_id == 5
    assert poly.start_validity_sec == 123.0
    assert poly.xypol_coeffs_ns == [1.0, 2.0, 3.0]
    assert poly.ypol_offset_ns == 4.0


def test_parse_delay_polynomial_from_mapping():
    data = {
        "start_validity_sec": 1.0,
        "validity_period_sec": 2.0,
        "xypol_coeffs_ns": [5.0],
        "ypol_offset_ns": 0.5,
    }
    poly = parse_delay_polynomial_from_attr_value(data, station_id=9)
    assert poly.station_id == 9
    assert poly.xypol_coeffs_ns == [5.0]


# ============================================================
# Integration: two sources, two feeds, genuinely different delay
# ============================================================


def test_two_sources_with_different_delay_feeds_diverge():
    """The whole point of per-source delay feeds: two tones in the SAME
    streamer, each fed a different DelayPolynomial (standing in for two
    different sky directions), must each receive their OWN, independently
    correct delay correction -- not the station's one shared delay."""
    station = StationConfig(
        station_id=1, substation_id=0, subarray_id=1, beam_id=1, first_channel_id=0, scan_id=1
    )
    feed_a = DelayFeed(name="source-a")
    feed_a.update(_poly(start_validity_sec=0.0, xypol_coeffs_ns=[0.0]))
    feed_b = DelayFeed(name="source-b")
    known_tau_ns = 5000.0
    feed_b.update(_poly(start_validity_sec=0.0, xypol_coeffs_ns=[known_tau_ns]))

    freq_a = sim.BASE_FREQ_HZ + 10 * CHANNEL_WIDTH_HZ + 150_000.0
    freq_b = sim.BASE_FREQ_HZ + 20 * CHANNEL_WIDTH_HZ + 150_000.0

    streamer = sim.DirectSynthesisStreamer(
        station=station,
        source_cfgs=[
            {"kind": "tone", "freq_hz": freq_a, "amplitude": 1.0, "delay_feed": feed_a},
            {"kind": "tone", "freq_hz": freq_b, "amplitude": 1.0, "delay_feed": feed_b},
        ],
        obs_time_ref=0.0,
        num_channels=32,
    )
    n = streamer.tick_n_samples()
    out = streamer.generate_next_tick(0.0, n)["V"]

    t = np.arange(n) / streamer.channel_output_rate
    residual_a = freq_a - (streamer.base_freq_hz + 10 * CHANNEL_WIDTH_HZ)
    residual_b = freq_b - (streamer.base_freq_hz + 20 * CHANNEL_WIDTH_HZ)

    expected_a = np.exp(1j * 2 * np.pi * residual_a * t)  # zero delay
    expected_b = np.exp(1j * 2 * np.pi * residual_b * t) * np.exp(
        -1j * 2 * np.pi * freq_b * known_tau_ns * 1e-9
    )

    assert np.max(np.abs(out[:, 10] - expected_a)) < 1e-9
    assert np.max(np.abs(out[:, 20] - expected_b)) < 1e-9

    # Confirms this isn't accidentally passing with source b's delay
    # silently defaulting to zero (e.g. a bug reintroducing one shared
    # feed): the zero-delay version of source b must NOT match.
    zero_delay_b = np.exp(1j * 2 * np.pi * residual_b * t)
    assert np.max(np.abs(out[:, 20] - zero_delay_b)) > 1e-3
