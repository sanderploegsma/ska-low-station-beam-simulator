"""
Tango device server entry point for the SPS station-beam simulator.

The actual signal-generation logic lives entirely in ``direct_synthesis.py``
(``DirectSynthesisStreamer``) — tone, per-pol station noise (a
pre-generated tile bank), and pulsed/pulsar sources are all handled by
direct, per-channel synthesis; there is no wideband+FFT fallback path.
Shared plumbing (delay polynomial, heap accumulation, SPEAD
packetization, the producer/sender loop) lives in ``common.py``, which
``direct_synthesis.py`` does not depend on beyond that shared plumbing.

PER-SCAN CONFIG: ``subarray_id``, ``beam_id``, and ``source_cfgs`` are no
longer device properties — they vary per scan (a station can be
reassigned between subarrays/beams across scans), so they're passed
dynamically as fields of the JSON object given to ``StartScan``. Only
``station_id``/``substation_id`` (identify the pod itself) and
``dest_ip``/``dest_port`` (the CBF endpoint) remain static device
properties.

PER-SOURCE DELAY: ``source_cfgs`` (in the ``StartScan`` JSON argument)
describes the tones/pulsars this station simulates. EVERY entry MUST name a
``delay_attr_uri`` — a Tango attribute on CBF's delay-poly emulator that
publishes CHANGE_EVENTs for that one source's direction (RA/Dec, Az/El,
or static; the emulator can expose several such directions, each on its
own attribute). There is deliberately no default delay for a source
missing one: ``DirectSynthesisStreamer`` refuses to construct a source
with no real delay path, since silently applying zero delay would
produce content that's trivially "perfectly aligned" and could mask a
real CBF delay-tracking bug rather than exercise it. ``StartScan``
subscribes to each named attribute and feeds its updates into a
``common.DelayFeed``, which ``DirectSynthesisStreamer`` then queries per
source, per tick, instead of every source sharing one station-level
delay.

UNVERIFIED, same caveat as the rest of this file's Tango-facing bits: the
exact attribute payload shape (see
``common.parse_delay_polynomial_from_attr_value``) and whether
``AttributeProxy`` delivers an immediate CHANGE_EVENT with the
attribute's current value on subscribe (rather than only on the next
actual change) both depend on how the real delay-poly emulator's
attributes are configured — confirm against it once available. Until a
first event arrives for a source, its ``DelayFeed`` applies zero delay
and logs a warning (see ``common.DelayFeed``) rather than blocking scan
start.
"""

from __future__ import annotations

import json
import queue
import threading

from tango import AttributeProxy, DevState, EventType
from tango.server import Device, attribute, command, device_property, run

from ska_low_station_beam_simulator.common import (
    QUEUE_MAXSIZE,
    ChannelHeap,
    DelayFeed,
    ScanRunner,
    SpsPacketizer,
    StationConfig,
    log,
    parse_delay_polynomial_from_attr_value,
    sender_loop,
)
from ska_low_station_beam_simulator.direct_synthesis import (
    DirectSynthesisStreamer,
    NoiseConfig,
    PulsarByNameConfig,
    PulsarByParamsConfig,
    SourceConfig,
    ToneSourceConfig,
)


def build_source_cfg(spec: dict, delay_feed: DelayFeed) -> SourceConfig:
    """Turns one ``source_cfgs`` JSON entry (plus its already-resolved
    ``DelayFeed``) into the typed config ``DirectSynthesisStreamer``
    expects -- the JSON-boundary equivalent of the type choice
    DirectSynthesisStreamer's direct Python callers make themselves (see
    direct_synthesis.py's SOURCE/NOISE CONFIG TYPES section). Kept as a
    standalone function, not inlined into ``StartScan``, so this JSON
    dispatch/validation logic is unit-testable without a live Tango
    device -- this codebase otherwise doesn't unit test the Tango device
    server layer at all (see CLAUDE.md's Setup section).

    :param spec: one raw ``source_cfgs`` entry (``'delay_attr_uri'``
        already consumed by the caller to build ``delay_feed`` -- an
        extra key here is harmless, ignored via ``dict`` unpacking).
    :param delay_feed: this source's already-subscribed ``DelayFeed``.
    :returns: a ``ToneSourceConfig``, ``PulsarByNameConfig``, or
        ``PulsarByParamsConfig``.
    :raises ValueError: for an unsupported ``kind``, or an ambiguous/
        incomplete pulsed source (both/neither of ``pulsar_name`` and
        ``period_s``/``width_s``/``dm_pc_cm3``).
    """
    cfg = dict(spec)
    cfg.pop("delay_attr_uri", None)
    kind = cfg.pop("kind", None)
    if kind == "tone":
        return ToneSourceConfig(delay_feed=delay_feed, **cfg)
    if kind == "pulsed":
        param_keys = ("period_s", "width_s", "dm_pc_cm3")
        has_name = "pulsar_name" in cfg
        present_params = [k for k in param_keys if k in cfg]
        if has_name and present_params:
            raise ValueError(
                f"pulsed source_cfgs entry has both 'pulsar_name' and "
                f"{present_params} -- specify one or the other, not both: "
                f"'pulsar_name' loads a pre-generated catalog entry, "
                f"'period_s'/'width_s'/'dm_pc_cm3' builds a custom "
                f"template at construction."
            )
        if has_name:
            return PulsarByNameConfig(delay_feed=delay_feed, **cfg)
        if len(present_params) == len(param_keys):
            return PulsarByParamsConfig(delay_feed=delay_feed, **cfg)
        missing = [k for k in param_keys if k not in cfg]
        raise ValueError(
            f"pulsed source_cfgs entry needs either 'pulsar_name' (load a "
            f"pre-generated catalog entry) or all of "
            f"'period_s'/'width_s'/'dm_pc_cm3' (build a custom template "
            f"at construction) -- got neither 'pulsar_name' nor {missing}."
        )
    raise ValueError(
        f"source_cfgs entry kind={kind!r} is not supported -- must be "
        f"'tone' or 'pulsed'."
    )


# ============================================================
# TANGO DEVICE SERVER
# ============================================================


class StationSimulatorDevice(Device):
    station_id = device_property(dtype=int, default_value=1)
    substation_id = device_property(dtype=int, default_value=0)
    dest_ip = device_property(dtype=str, default_value="127.0.0.1")
    dest_port = device_property(dtype=int, default_value=8000)

    def init_device(self):
        super().init_device()
        self._send_queue: queue.Queue[ChannelHeap] = queue.Queue(maxsize=QUEUE_MAXSIZE)
        self._shutdown_event = threading.Event()
        self._scan_runner: ScanRunner | None = None
        self._delay_subscriptions: list[tuple[AttributeProxy, int]] = []

        # subarray_id/beam_id are unknown until the first StartScan --
        # placeholder 0s here, overwritten (on the same StationConfig
        # instance, which SpsPacketizer holds a live reference to) each
        # StartScan call.
        self._station_cfg = StationConfig(
            station_id=self.station_id,
            substation_id=self.substation_id,
            subarray_id=0,
            beam_id=0,
        )
        self._packetizer = SpsPacketizer(
            self._station_cfg, self.dest_ip, self.dest_port
        )

        self._sender_thread = threading.Thread(
            target=sender_loop,
            args=(self._send_queue, self._packetizer, self._shutdown_event),
            daemon=True,
        )
        self._sender_thread.start()
        self.set_state(DevState.ON)

    def _make_delay_feed(self, attr_uri: str) -> DelayFeed:
        """Subscribes to ``attr_uri``'s CHANGE_EVENTs and returns a
        ``DelayFeed`` that always reflects the most recently pushed
        value. The subscription itself is torn down in
        ``_teardown_delay_subscriptions`` (called from ``StartScan``
        before setting up the next scan's subscriptions, and from
        ``StopScan``/``delete_device``) — never left dangling across
        scans.

        :param attr_uri: the Tango attribute to subscribe to.
        :returns: a ``DelayFeed`` that ``DirectSynthesisStreamer`` can
            query for this source's current delay polynomial.
        """
        feed = DelayFeed(name=attr_uri)
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
            feed.update(poly)

        event_id = proxy.subscribe_event(EventType.CHANGE_EVENT, _on_event)
        self._delay_subscriptions.append((proxy, event_id))
        return feed

    def _teardown_delay_subscriptions(self):
        for proxy, event_id in self._delay_subscriptions:
            try:
                proxy.unsubscribe_event(event_id)
            except Exception:  # noqa: BLE001
                log.exception("failed to unsubscribe from a delay-poly attribute")
        self._delay_subscriptions = []

    @command(
        dtype_in=str,
        doc_in=(
            "JSON object: {obs_time_epoch_s, scan_duration_s, scan_id, "
            "subarray_id, beam_id, source_cfgs}. source_cfgs is a JSON "
            "list (see direct_synthesis.DirectSynthesisStreamer) -- EVERY "
            "entry MUST include 'delay_attr_uri' naming a Tango attribute "
            "on CBF's delay-poly emulator to subscribe for that source's "
            "own delay polynomial (there is no default delay -- see "
            "module docstring). An empty list means no tone/pulsar "
            "sources at all for this scan (noise, if noise_cfg is set, "
            "still plays). A 'pulsed' entry may give either "
            "'pulsar_name' (loads a pre-generated catalog entry -- fast "
            "startup, fixed parameters, see pulsar_catalog.py) or "
            "'period_s'/'width_s'/'dm_pc_cm3' (builds a custom template "
            "at construction -- arbitrary parameters, slower startup), "
            "never both."
        ),
    )
    def StartScan(self, args_json):
        args = json.loads(args_json)
        obs_time = args["obs_time_epoch_s"]
        scan_duration_s = args["scan_duration_s"]
        scan_id = args["scan_id"]
        if self._scan_runner is not None and self._scan_runner.thread.is_alive():
            raise RuntimeError("scan already running — call StopScan first")

        self._station_cfg.subarray_id = int(args["subarray_id"])
        self._station_cfg.beam_id = int(args["beam_id"])
        self._station_cfg.scan_id = int(scan_id)

        source_specs = args.get("source_cfgs", [])
        noise_cfg = NoiseConfig(std=0.05, seed=self.station_id)

        self._teardown_delay_subscriptions()
        source_cfgs: list[SourceConfig] = []
        for spec in source_specs:
            attr_uri = spec.get("delay_attr_uri")
            if not attr_uri:
                raise ValueError(
                    f"source_cfgs entry kind={spec.get('kind')!r} is "
                    f"missing required 'delay_attr_uri' — every source must "
                    f"name a delay-poly attribute to subscribe to, there is "
                    f"no default delay (see module docstring)."
                )
            delay_feed = self._make_delay_feed(attr_uri)
            source_cfgs.append(build_source_cfg(spec, delay_feed))

        streamer = DirectSynthesisStreamer(
            station=self._station_cfg,
            source_cfgs=source_cfgs,
            noise_cfg=noise_cfg,
            obs_time_ref=obs_time,
        )
        self._scan_runner = ScanRunner(
            streamer=streamer,
            send_queue=self._send_queue,
            obs_time=obs_time,
            scan_duration_s=scan_duration_s,
        )
        self._scan_runner.start()
        self.set_state(DevState.RUNNING)

    @command
    def StopScan(self):
        if self._scan_runner is not None:
            self._scan_runner.stop()
        self._teardown_delay_subscriptions()
        self.set_state(DevState.ON)

    @attribute(dtype=int)
    def queue_depth(self):
        return self._send_queue.qsize()

    def delete_device(self):
        if self._scan_runner is not None:
            self._scan_runner.stop()
        self._teardown_delay_subscriptions()
        self._shutdown_event.set()
        self._sender_thread.join(timeout=5.0)
        super().delete_device()


if __name__ == "__main__":
    run((StationSimulatorDevice,))
