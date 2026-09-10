"""
Tango device server entry point for the SPS station-beam simulator.

This device no longer generates SPEAD/UDP content itself. Signal
generation, heap accumulation, and SPEAD/UDP sending have moved to a
separate Go process (see the repo root's ``cmd/server``,
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
``cmd/server/main.go``) and must already agree with this device's
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
import threading

import grpc
import ska_tango_base.future as stb
from ska_control_model import HealthState
from tango import AttributeProxy, EnsureOmniThread, EventType
from tango.server import command, device_property, run

from ska_low_station_beam_simulator.common import (
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


class StationBeamSimulator(stb.BaseInterface):
    station_id: int = device_property(default_value=1)  # type: ignore[assignment]
    substation_id: int = device_property(default_value=0)  # type: ignore[assignment]
    # host:port of the Go gRPC simulator this device drives -- see
    # cmd/server's -listen flag (default matches here). Must already
    # be running with the SAME station_id/substation_id and the real CBF
    # dest_ip/dest_port (its own CLI flags now, not device properties on
    # this side -- see module docstring).
    grpc_target: str = device_property(default_value="localhost:50051")  # type: ignore[assignment]

    # How often WatchStatus pushes a status update, consumed by the
    # background thread that feeds scan_running/queue_depth/
    # drift_seconds/tick_number below. <=0 uses the Go server's own
    # default (see WatchStatus's doc comment in api/simulator.proto).
    status_update_interval_s: float = device_property(default_value=1.0)  # type: ignore[assignment]

    scan_running_signal = stb.Signal[bool](stored=True)
    queue_depth_signal = stb.CachingAttrSignal[int]()
    drift_seconds_signal = stb.CachingAttrSignal[float]()
    tick_number_signal = stb.CachingAttrSignal[int]()

    def init_device(self):
        super().init_device()
        self._delay_subscriptions: list[tuple[AttributeProxy, int]] = []
        self._status_lock = threading.Lock()
        self._last_status: simulator_pb2.StatusResponse | None = None

        self.logger.info("Connecting to gRPC endpoint %s", self.grpc_target)
        self._channel = grpc.insecure_channel(self.grpc_target)
        self._stub = simulator_pb2_grpc.StationSimulatorStub(self._channel)

        # WatchStatus is a separate, long-lived RPC from the GetStatus
        # call just above -- it lives for this device's whole lifetime
        # (across StartScan/StopScan calls, not scoped to one scan), and
        # gRPC multiplexes it on the same channel as ordinary unary
        # calls, so it never blocks StartScan/StopScan/PushDelayUpdate.
        self._watch_stop = threading.Event()
        self._watch_thread = threading.Thread(
            target=self._watch_status_loop,
            name="WatchStatus",
        )
        self._watch_thread.start()

        self.init_completed()

    def delete_device(self):
        try:
            self._stub.StopScan(simulator_pb2.StopScanRequest())
        except grpc.RpcError:
            self.logger.exception("StopScan request failed during delete_device")
        self._teardown_delay_subscriptions()
        # Signal the watcher first, then close the channel -- that's
        # what actually unblocks WatchStatus's blocking stream iterator
        # in _watch_status_loop, letting the thread notice _watch_stop
        # and exit instead of retrying against a closed channel.
        with self.allow_internal_threads():
            self._watch_stop.set()
            self._channel.close()
            self._watch_thread.join(timeout=5.0)
        super().delete_device()

    def _watch_status_loop(self) -> None:
        """Background consumer of the WatchStatus stream -- kept
        alongside on-demand GetStatus calls (``_fetch_status``), not a
        replacement for the RPC itself: this is what lets a single
        long-lived stream serve all four status attributes below
        instead of each attribute read triggering its own independent
        GetStatus round trip (see docs/history.md's note on that being
        a deliberate-but-wasteful simplification).

        Retries with a fixed backoff on any RPC error (e.g. the Go
        process restarting) -- reporting HealthState.FAILED for the
        duration, exactly as ``_fetch_status`` already does for GetStatus
        failures -- rather than giving up permanently.
        """
        with EnsureOmniThread():
            backoff_s = 2.0
            while not self._watch_stop.is_set():
                try:
                    stream = self._stub.WatchStatus(
                        simulator_pb2.WatchStatusRequest(
                            update_interval_s=self.status_update_interval_s
                        )
                    )
                    for response in stream:
                        self._update_status(response)
                        if self.healthState != HealthState.OK:
                            self.report_health(HealthState.OK, [])
                        if self._watch_stop.is_set():
                            stream.cancel()
                            break
                except grpc.RpcError as e:
                    if self._watch_stop.is_set():
                        break
                    self.logger.warning("WatchStatus stream failed, retrying: %s", e)
                    if self.healthState != HealthState.FAILED:
                        self.report_health(
                            HealthState.FAILED,
                            [f"Unable to connect to {self.grpc_target}", str(e)],
                        )
                    self._watch_stop.wait(backoff_s)

    def _update_status(self, response: simulator_pb2.StatusResponse) -> None:
        self.scan_running_signal = response.scan_running
        self.queue_depth_signal = response.queue_depth
        self.drift_seconds_signal = response.drift_seconds
        self.tick_number_signal = response.tick_number

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
                self.logger.warning(
                    "delay-poly attribute event error for %s: %s",
                    attr_uri,
                    event.errors,
                )
                return
            try:
                poly = parse_delay_polynomial_from_attr_value(
                    event.attr_value.value, self.station_id
                )
            except Exception:
                self.logger.exception(
                    "failed to parse delay polynomial pushed by %s", attr_uri
                )
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
                self.logger.exception(
                    "failed to forward delay-poly update for %s to the gRPC simulator",
                    attr_uri,
                )

        self.logger.info("Subscribing to delay-poly attribute %s", proxy.name())
        event_id = proxy.subscribe_event(EventType.CHANGE_EVENT, _on_event)
        self._delay_subscriptions.append((proxy, event_id))

    def _teardown_delay_subscriptions(self):
        for proxy, event_id in self._delay_subscriptions:
            self.logger.info("Unsubscribing from delay-poly attribute %s", proxy.name())
            try:
                proxy.unsubscribe_event(event_id)
            except Exception:
                self.logger.exception(
                    "failed to unsubscribe from delay-poly attribute %s", proxy.name()
                )
        self._delay_subscriptions = []

    scan_running = stb.attribute_from_signal(scan_running_signal)
    queue_depth = stb.attribute_from_signal(queue_depth_signal)
    drift_seconds = stb.attribute_from_signal(drift_seconds_signal)
    tick_number = stb.attribute_from_signal(tick_number_signal)

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
        if self.scan_running_signal:  # type: ignore
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
        self.logger.info(
            "Starting scan %s with %d tone sources, obs_time=%s, duration=%s",
            scan_request.scan_id,
            len(scan_request.tone_sources),
            scan_request.obs_time_epoch_s,
            scan_request.scan_duration_s,
        )
        try:
            response = self._stub.StartScan(scan_request)
        except grpc.RpcError as e:
            self._teardown_delay_subscriptions()
            self.logger.exception("StartScan request failed")
            raise RuntimeError(f"StartScan request failed: {e.details()}") from e
        if not response.ok:
            self._teardown_delay_subscriptions()
            msg = f"StartScan request rejected: {response.message}"
            self.logger.warning(msg)
            raise RuntimeError(msg)

    @command
    def StopScan(self):
        self.logger.info("Stopping current scan")
        try:
            self._stub.StopScan(simulator_pb2.StopScanRequest())
        except grpc.RpcError:
            self.logger.exception("StopScan request failed")
        self._teardown_delay_subscriptions()


if __name__ == "__main__":
    run((StationBeamSimulator,))
