"""Tests for StationSimulatorDevice's delay-poly attribute subscription
wiring (_make_delay_feed / _teardown_delay_subscriptions).

Doesn't stand up a real Tango device server (this codebase doesn't unit
test that layer anywhere else either — see CLAUDE.md's Setup section:
pytango degrades simulator.py to stub classes if unavailable, but
StationSimulatorDevice itself still isn't deployable without a live
Tango context). Instead, `AttributeProxy`/`EventType` are monkeypatched
with fakes so the actual new logic here -- subscribe/unsubscribe
pairing, event-error handling, parse-failure handling -- can be
exercised directly against a lightweight stand-in object.
"""

import types

import pytest

from ska_low_station_beam_simulator import simulator as sim


class _FakeEvent:
    def __init__(self, value=None, err=False, errors=None):
        self.err = err
        self.errors = errors or []
        self.attr_value = types.SimpleNamespace(value=value)


class _FakeAttributeProxy:
    instances = []

    def __init__(self, attr_uri):
        self.attr_uri = attr_uri
        self.subscriptions = {}
        self._next_id = 1
        self.unsubscribed = []
        _FakeAttributeProxy.instances.append(self)

    def subscribe_event(self, event_type, callback):
        event_id = self._next_id
        self._next_id += 1
        self.subscriptions[event_id] = callback
        return event_id

    def unsubscribe_event(self, event_id):
        self.unsubscribed.append(event_id)
        self.subscriptions.pop(event_id, None)


@pytest.fixture(autouse=True)
def _fake_attribute_proxy(monkeypatch):
    _FakeAttributeProxy.instances = []
    monkeypatch.setattr(sim, "AttributeProxy", _FakeAttributeProxy)
    monkeypatch.setattr(sim, "EventType", types.SimpleNamespace(CHANGE_EVENT="CHANGE_EVENT"))


def _fake_device():
    """A bare stand-in with just the attributes/state
    _make_delay_feed/_teardown_delay_subscriptions actually touch —
    avoids needing a live Tango device server context."""
    dev = types.SimpleNamespace()
    dev.station_id = 1
    dev._delay_subscriptions = []
    return dev


VALID_PAYLOAD = (
    '{"start_validity_sec": 100.0, "validity_period_sec": 600.0, '
    '"xypol_coeffs_ns": [750.0], "ypol_offset_ns": 2.0}'
)


def test_make_delay_feed_subscribes_and_applies_pushed_value():
    dev = _fake_device()
    feed = sim.StationSimulatorDevice._make_delay_feed(dev, "sys/delaypoly/1/direction0")

    assert len(dev._delay_subscriptions) == 1
    proxy, event_id = dev._delay_subscriptions[0]
    assert proxy.attr_uri == "sys/delaypoly/1/direction0"

    proxy.subscriptions[event_id](_FakeEvent(value=VALID_PAYLOAD))
    poly = feed.get(100.0)
    assert poly.xypol_coeffs_ns == [750.0]
    assert poly.ypol_offset_ns == 2.0
    assert poly.station_id == dev.station_id


def test_make_delay_feed_ignores_error_events(caplog):
    dev = _fake_device()
    feed = sim.StationSimulatorDevice._make_delay_feed(dev, "sys/delaypoly/1/direction0")
    proxy, event_id = dev._delay_subscriptions[0]

    proxy.subscriptions[event_id](_FakeEvent(err=True, errors=["boom"]))

    # no update was ever applied -- still on the zero-delay default
    poly = feed.get(0.0)
    assert sum(poly.xypol_coeffs_ns) == 0.0


def test_make_delay_feed_survives_unparseable_payload():
    dev = _fake_device()
    feed = sim.StationSimulatorDevice._make_delay_feed(dev, "sys/delaypoly/1/direction0")
    proxy, event_id = dev._delay_subscriptions[0]

    proxy.subscriptions[event_id](_FakeEvent(value="not valid json"))

    poly = feed.get(0.0)
    assert sum(poly.xypol_coeffs_ns) == 0.0  # malformed push is dropped, not crashed on

    # a later, valid push still works normally
    proxy.subscriptions[event_id](_FakeEvent(value=VALID_PAYLOAD))
    assert feed.get(100.0).xypol_coeffs_ns == [750.0]


def test_teardown_unsubscribes_all_and_clears_list():
    dev = _fake_device()
    sim.StationSimulatorDevice._make_delay_feed(dev, "sys/delaypoly/1/direction0")
    sim.StationSimulatorDevice._make_delay_feed(dev, "sys/delaypoly/1/direction1")
    assert len(dev._delay_subscriptions) == 2

    proxies = [p for p, _ in dev._delay_subscriptions]
    sim.StationSimulatorDevice._teardown_delay_subscriptions(dev)

    assert dev._delay_subscriptions == []
    for proxy in proxies:
        assert len(proxy.unsubscribed) == 1


def test_teardown_survives_unsubscribe_failure():
    dev = _fake_device()

    class _BrokenAttributeProxy(_FakeAttributeProxy):
        def unsubscribe_event(self, event_id):
            raise RuntimeError("connection lost")

    dev._delay_subscriptions = [(_BrokenAttributeProxy("x"), 1), (_FakeAttributeProxy("y"), 1)]
    # must not raise even though the first proxy's unsubscribe fails
    sim.StationSimulatorDevice._teardown_delay_subscriptions(dev)
    assert dev._delay_subscriptions == []
