"""
Tango device server entry point for the SPS station-beam simulator.

This device no longer generates SPEAD/UDP content itself. Signal
generation, heap accumulation, and SPEAD/UDP sending have moved to a
separate Go process (see the repo root's ``cmd/simulator``,
``api/simulator.proto``) that this device drives over gRPC — this
module now owns only the Tango-facing bits: device properties,
StartScan/StopScan command handling, and subscribing to CBF's
delay-poly emulator's CHANGE_EVENTs and forwarding them on via
PushDelayUpdate. This is exactly the split ``api/simulator.proto``'s own
doc comment describes (a Tango device server owns Tango, this Go
process owns signal generation + SPEAD/UDP sending, with no Tango access
of its own).

PROTOTYPE SCOPE, inherited from the Go side (see ``api/simulator.proto``):
only ``'tone'`` ``source_cfgs`` entries (plus noise) are supported
through this gRPC path — a ``'pulsed'`` (pulsar) entry raises
immediately from ``build_tone_source_request``, rather than silently
doing nothing or falling back to a local Python generation path (there
is no local generation path left in this device at all).
``direct_synthesis.py``'s pulsar support is untouched and still directly
usable (``benchmark_direct_synthesis.py``, ``generate_test_pcap.py``,
``tests/test_direct_synthesis.py``) — only this Tango-facing device no
longer drives it.

PER-SCAN CONFIG: unchanged from before — ``subarray_id``, ``beam_id``,
and ``source_cfgs`` arrive per ``StartScan`` call (a station can be
reassigned between subarrays/beams across scans), not as device
properties. ``station_id``/``substation_id`` remain static device
properties, used to tag delay-poly pushes with this station's ID (see
``parse_delay_polynomial_from_attr_value``) — they are NOT forwarded to
the Go process over gRPC, which gets its own station_id/substation_id
independently at deploy time (its own CLI flags, see
``cmd/simulator/main.go``) and must already agree with this device's
values. ``dest_ip``/``dest_port`` are gone from this device entirely:
the Go process owns the CBF SPEAD/UDP destination now (also its own CLI
flags), not this device.

PER-SOURCE DELAY: unchanged in spirit — every ``'tone'`` ``source_cfgs``
entry MUST name a ``delay_attr_uri``, a Tango attribute on CBF's
delay-poly emulator publishing CHANGE_EVENTs for that source's
direction. This device still owns the ``AttributeProxy`` subscription
(Go has no Tango access of its own), but instead of feeding a local
``common.DelayFeed`` consumed by a local streamer, a pushed update is
now translated into a ``DelayPolynomial`` protobuf message and forwarded
via ``PushDelayUpdate``, keyed by ``source_id``. This device reuses each
source's own ``delay_attr_uri`` AS its ``source_id`` (already required,
already unique per source by construction — attempting to reuse one
``delay_attr_uri`` across two sources in the same scan is rejected by
the gRPC server as a duplicate ``source_id``), so no new JSON field is
needed for it.

UNVERIFIED, same caveat as before: the exact attribute payload shape
(see ``common.parse_delay_polynomial_from_attr_value``) and whether
``AttributeProxy`` delivers an immediate CHANGE_EVENT with the
attribute's current value on subscribe both depend on how the real
delay-poly emulator is configured — confirm against it once available.
"""

from __future__ import annotations

import json

import grpc
from tango import AttributeProxy, DevState, EventType
from tango.server import Device, attribute, command, device_property, run

from ska_low_station_beam_simulator.common import (
    log,
    parse_delay_polynomial_from_attr_value,
)
from ska_low_station_beam_simulator.simulatorpb import (
    simulator_pb2,
    simulator_pb2_grpc,
)


def build_tone_source_request(spec: dict) -> simulator_pb2.ToneSourceConfig:
    """Turns one ``'tone'`` ``source_cfgs`` JSON entry into the
    ``ToneSourceConfig`` protobuf message ``StartScan`` sends to the Go
    gRPC simulator — validated at the point untyped data enters the
    system. Kept standalone, not inlined into ``StartScan``, so it's
    unit-testable without a live Tango device or a live gRPC server.

    :param spec: one raw ``source_cfgs`` entry.
    :returns: a ``ToneSourceConfig`` with ``source_id`` set to
        ``spec['delay_attr_uri']`` (consumed here, not forwarded as a
        stray field — the caller subscribes to that same URI, see
        ``StartScan``/``_make_delay_feed``).
    :raises ValueError: if ``spec['kind']`` isn't ``'tone'`` (the gRPC
        backend doesn't support pulsed/pulsar sources yet — see module
        docstring) or ``'delay_attr_uri'`` is missing (there is no
        default delay).
    """
    kind = spec.get("kind")
    if kind != "tone":
        raise ValueError(
            f"source_cfgs entry kind={kind!r} is not supported by the "
            f"gRPC simulator backend -- only 'tone' sources (plus noise) "
            f"are implemented there so far (see api/simulator.proto's "
            f"doc comment)."
        )
    cfg = dict(spec)
    cfg.pop("kind")
    attr_uri = cfg.pop("delay_attr_uri", None)
    if not attr_uri:
        raise ValueError(
            "source_cfgs entry is missing required 'delay_attr_uri' -- "
            "every source must name a delay-poly attribute to subscribe "
            "to, there is no default delay (see module docstring)."
        )
    return simulator_pb2.ToneSourceConfig(source_id=attr_uri, **cfg)


# ============================================================
# TANGO DEVICE SERVER
# ============================================================


class StationSimulatorDevice(Device):
    station_id = device_property(dtype=int, default_value=1)
    substation_id = device_property(dtype=int, default_value=0)
    # host:port of the Go gRPC simulator this device drives -- see
    # cmd/simulator's -listen flag (default matches here). Must already
    # be running with the SAME station_id/substation_id and the real CBF
    # dest_ip/dest_port (its own CLI flags now, not device properties on
    # this side -- see module docstring).
    grpc_target = device_property(dtype=str, default_value="localhost:50051")

    def init_device(self):
        super().init_device()
        self._delay_subscriptions: list[tuple[AttributeProxy, int]] = []
        # insecure_channel doesn't dial until the first RPC -- a
        # misconfigured/unreachable grpc_target only surfaces once
        # StartScan (or an attribute read) actually calls out, not here.
        self._channel = grpc.insecure_channel(self.grpc_target)
        self._stub = simulator_pb2_grpc.StationSimulatorStub(self._channel)
        self.set_state(DevState.ON)

    def _make_delay_feed(self, attr_uri: str) -> None:
        """Subscribes to ``attr_uri``'s CHANGE_EVENTs and forwards every
        pushed value to the gRPC simulator via ``PushDelayUpdate``, keyed
        by ``source_id=attr_uri`` (see module docstring for why the
        attribute URI doubles as the source_id). The subscription itself
        is torn down in ``_teardown_delay_subscriptions`` (called from
        ``StartScan`` before setting up the next scan's subscriptions,
        and from ``StopScan``/``delete_device``) — never left dangling
        across scans.

        :param attr_uri: the Tango attribute to subscribe to; also this
            source's gRPC ``source_id``.
        """
        proxy = AttributeProxy(attr_uri)

        def _on_event(event):
            if event.err:
                log.warning(
                    "delay-poly attribute event error for %s: %s",
                    attr_uri,
                    event.errors,
                )
                return
            try:
                poly = parse_delay_polynomial_from_attr_value(
                    event.attr_value.value, self.station_id
                )
            except Exception:  # noqa: BLE001
                log.exception("failed to parse delay polynomial pushed by %s", attr_uri)
                return
            try:
                self._stub.PushDelayUpdate(
                    simulator_pb2.PushDelayUpdateRequest(
                        source_id=attr_uri,
                        polynomial=simulator_pb2.DelayPolynomial(
                            station_id=poly.station_id,
                            start_validity_sec=poly.start_validity_sec,
                            validity_period_sec=poly.validity_period_sec,
                            xypol_coeffs_ns=poly.xypol_coeffs_ns,
                            ypol_offset_ns=poly.ypol_offset_ns,
                        ),
                    )
                )
            except grpc.RpcError:
                log.exception(
                    "failed to forward delay-poly update for %s to the gRPC simulator",
                    attr_uri,
                )

        event_id = proxy.subscribe_event(EventType.CHANGE_EVENT, _on_event)
        self._delay_subscriptions.append((proxy, event_id))

    def _teardown_delay_subscriptions(self):
        for proxy, event_id in self._delay_subscriptions:
            try:
                proxy.unsubscribe_event(event_id)
            except Exception:  # noqa: BLE001
                log.exception("failed to unsubscribe from a delay-poly attribute")
        self._delay_subscriptions = []

    def _get_status(self) -> simulator_pb2.StatusResponse:
        return self._stub.GetStatus(simulator_pb2.GetStatusRequest())

    @command(
        dtype_in=str,
        doc_in=(
            "JSON object: {obs_time_epoch_s, scan_duration_s, scan_id, "
            "subarray_id, beam_id, source_cfgs}. source_cfgs is a JSON "
            "list -- EVERY entry MUST include 'delay_attr_uri' naming a "
            "Tango attribute on CBF's delay-poly emulator to subscribe "
            "for that source's own delay polynomial (there is no "
            "default delay -- see module docstring), and MUST have "
            "kind='tone' (the gRPC simulator backend doesn't support "
            "'pulsed' sources yet). An empty list means no tone sources "
            "at all for this scan (noise still plays)."
        ),
    )
    def StartScan(self, args_json):
        args = json.loads(args_json)

        # Checked up front, before touching any subscription, so a
        # rejected StartScan (scan already running) never tears down the
        # ACTUAL running scan's delay subscriptions -- see StartScan's
        # gRPC-error handling below for the remaining (pre-existing,
        # equally non-atomic) check-then-act race with a concurrent
        # StartScan call, which the Go server's own FailedPrecondition
        # check is the real backstop for.
        try:
            already_running = self._get_status().scan_running
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC GetStatus failed: {e.details()}") from e
        if already_running:
            raise RuntimeError("scan already running — call StopScan first")

        source_specs = args.get("source_cfgs", [])
        self._teardown_delay_subscriptions()
        tone_sources = []
        for spec in source_specs:
            request = build_tone_source_request(spec)
            self._make_delay_feed(request.source_id)
            tone_sources.append(request)

        scan_request = simulator_pb2.StartScanRequest(
            obs_time_epoch_s=args["obs_time_epoch_s"],
            scan_duration_s=args["scan_duration_s"],
            scan_id=int(args["scan_id"]),
            subarray_id=int(args["subarray_id"]),
            beam_id=int(args["beam_id"]),
            tone_sources=tone_sources,
            noise=simulator_pb2.NoiseConfig(std=0.05, seed=self.station_id),
        )
        try:
            response = self._stub.StartScan(scan_request)
        except grpc.RpcError as e:
            self._teardown_delay_subscriptions()
            raise RuntimeError(f"gRPC StartScan failed: {e.details()}") from e
        if not response.ok:
            self._teardown_delay_subscriptions()
            raise RuntimeError(f"gRPC StartScan rejected: {response.message}")
        self.set_state(DevState.RUNNING)

    @command
    def StopScan(self):
        try:
            self._stub.StopScan(simulator_pb2.StopScanRequest())
        except grpc.RpcError:
            log.exception("gRPC StopScan failed")
        self._teardown_delay_subscriptions()
        self.set_state(DevState.ON)

    @attribute(dtype=int)
    def queue_depth(self):
        return self._get_status().queue_depth

    @attribute(dtype=float)
    def drift_seconds(self):
        return self._get_status().drift_seconds

    @attribute(dtype=int)
    def tick_number(self):
        return self._get_status().tick_number

    def delete_device(self):
        try:
            self._stub.StopScan(simulator_pb2.StopScanRequest())
        except grpc.RpcError:
            log.exception("gRPC StopScan failed during delete_device")
        self._teardown_delay_subscriptions()
        self._channel.close()
        super().delete_device()


if __name__ == "__main__":
    run((StationSimulatorDevice,))
