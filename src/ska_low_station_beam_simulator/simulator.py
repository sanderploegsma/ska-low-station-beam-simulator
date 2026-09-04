"""
Tango device server entry point for the SPS station-beam simulator.

The actual signal-generation logic lives entirely in direct_synthesis.py
(DirectSynthesisStreamer) — tone, per-pol station noise (a pre-generated
tile bank), and pulsed/pulsar sources are all handled by direct,
per-channel synthesis; there is no wideband+FFT fallback path anymore
(the legacy wideband_streamer.py/StationStreamer this simulator used to
fall back to for pulsed sources was removed once direct_synthesis.py
gained a direct per-channel pulsar representation — see CLAUDE.md).
Shared plumbing (delay polynomial, heap accumulation, SPEAD packetization,
the producer/sender loop) lives in common.py, which direct_synthesis.py
does not depend on beyond that shared plumbing.

PER-SOURCE DELAY: `source_cfgs_json` (a device_property) describes the
tones/pulsars this station simulates. EVERY entry MUST name a
`delay_attr_uri` — a Tango attribute on CBF's delay-poly emulator that
publishes CHANGE_EVENTs for that one source's direction (RA/Dec, Az/El,
or static; the emulator can expose several such directions, each on its
own attribute). There is deliberately no default delay for a source
missing one: DirectSynthesisStreamer refuses to construct a source with
no real delay path, since silently applying zero delay would produce
content that's trivially "perfectly aligned" and could mask a real CBF
delay-tracking bug rather than exercise it. StartScan subscribes to each
named attribute and feeds its updates into a common.DelayFeed, which
DirectSynthesisStreamer then queries per source, per tick, instead of
every source sharing one station-level delay.

UNVERIFIED, same caveat as the rest of this file's Tango-facing bits: the
exact attribute payload shape (see
common.parse_delay_polynomial_from_attr_value) and whether AttributeProxy
delivers an immediate CHANGE_EVENT with the attribute's current value on
subscribe (rather than only on the next actual change) both depend on how
the real delay-poly emulator's attributes are configured — confirm
against it once available. Until a first event arrives for a source, its
DelayFeed applies zero delay and logs a warning (see common.DelayFeed)
rather than blocking scan start.
"""

from __future__ import annotations

import json
import queue
import threading
from typing import Optional

from ska_low_station_beam_simulator.common import (
    ChannelHeap,
    QUEUE_MAXSIZE,
    DelayFeed,
    ScanRunner,
    SpsPacketizer,
    StationConfig,
    log,
    parse_delay_polynomial_from_attr_value,
    sender_loop,
)
from ska_low_station_beam_simulator.direct_synthesis import DirectSynthesisStreamer

try:
    from tango import AttributeProxy, DevState, EventType
    from tango.server import Device, attribute, command, device_property, run

    TANGO_AVAILABLE = True
except ImportError:
    TANGO_AVAILABLE = False
    log.warning(
        "pytango not installed — device server class will be importable "
        "but NOT deployable."
    )

    class DevState:
        ON = "ON"
        RUNNING = "RUNNING"

    class EventType:
        CHANGE_EVENT = "CHANGE_EVENT"

    class AttributeProxy:  # pragma: no cover - stub, not deployable
        def __init__(self, *_a, **_kw):
            raise RuntimeError("pytango is required to subscribe to a delay-poly attribute")

    def command(*_a, **_kw):
        def deco(f):
            return f

        return deco

    def attribute(*_a, **_kw):
        def deco(f):
            return f

        return deco

    def device_property(*_a, **_kw):
        return None

    class Device:
        def set_state(self, *_a, **_kw):
            pass

        def init_device(self):
            pass


# ============================================================
# TANGO DEVICE SERVER
# ============================================================


class StationSimulatorDevice(Device):
    station_id = device_property(dtype=int, default_value=1)
    substation_id = device_property(dtype=int, default_value=0)
    subarray_id = device_property(dtype=int, default_value=1)
    beam_id = device_property(dtype=int, default_value=1)
    # 64 = confirmed GLOBAL coarse channel ID of BASE_FREQ_HZ (50.0 MHz,
    # the lowest valid SKA-Low channel centre) -- see common.StationConfig.
    first_channel_id = device_property(dtype=int, default_value=64)
    dest_ip = device_property(dtype=str, default_value="127.0.0.1")
    dest_port = device_property(dtype=int, default_value=8000)

    # JSON list of source_cfgs (see direct_synthesis.DirectSynthesisStreamer)
    # -- EVERY entry MUST include "delay_attr_uri" naming a Tango attribute
    # on CBF's delay-poly emulator to subscribe for that source's own
    # delay polynomial (there is no default delay -- see module
    # docstring). Empty ("[]", the default) means no tone/pulsar sources
    # at all for this scan (noise, if noise_cfg is set, still plays).
    source_cfgs_json = device_property(dtype=str, default_value="[]")

    def init_device(self):
        super().init_device()
        self._send_queue: "queue.Queue[ChannelHeap]" = queue.Queue(
            maxsize=QUEUE_MAXSIZE
        )
        self._shutdown_event = threading.Event()
        self._scan_runner: Optional[ScanRunner] = None
        self._delay_subscriptions: list[tuple["AttributeProxy", int]] = []

        self._station_cfg = StationConfig(
            station_id=self.station_id,
            substation_id=self.substation_id,
            subarray_id=self.subarray_id,
            beam_id=self.beam_id,
            first_channel_id=self.first_channel_id,
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
        """Subscribes to `attr_uri`'s CHANGE_EVENTs and returns a
        DelayFeed that always reflects the most recently pushed value.
        The subscription itself is torn down in
        _teardown_delay_subscriptions (called from StartScan before
        setting up the next scan's subscriptions, and from StopScan/
        delete_device) — never left dangling across scans."""
        feed = DelayFeed(name=attr_uri)
        proxy = AttributeProxy(attr_uri)

        def _on_event(event):
            if event.err:
                log.warning(
                    "delay-poly attribute event error for %s: %s",
                    attr_uri, event.errors,
                )
                return
            try:
                poly = parse_delay_polynomial_from_attr_value(
                    event.attr_value.value, self.station_id
                )
            except Exception:
                log.exception(
                    "failed to parse delay polynomial pushed by %s", attr_uri
                )
                return
            feed.update(poly)

        event_id = proxy.subscribe_event(EventType.CHANGE_EVENT, _on_event)
        self._delay_subscriptions.append((proxy, event_id))
        return feed

    def _teardown_delay_subscriptions(self):
        for proxy, event_id in self._delay_subscriptions:
            try:
                proxy.unsubscribe_event(event_id)
            except Exception:
                log.exception("failed to unsubscribe from a delay-poly attribute")
        self._delay_subscriptions = []

    @command(dtype_in=[float], doc_in="[obs_time_epoch_s, scan_duration_s, scan_id]")
    def StartScan(self, args):
        obs_time, scan_duration_s, scan_id = args
        if self._scan_runner is not None and self._scan_runner.thread.is_alive():
            raise RuntimeError("scan already running — call StopScan first")

        self._station_cfg.scan_id = int(scan_id)

        source_specs = json.loads(self.source_cfgs_json) if self.source_cfgs_json else []
        noise_cfg = {"std": 0.05, "seed": self.station_id}

        self._teardown_delay_subscriptions()
        source_cfgs = []
        for spec in source_specs:
            cfg = dict(spec)
            attr_uri = cfg.pop("delay_attr_uri", None)
            if not attr_uri:
                raise ValueError(
                    f"source_cfgs_json entry kind={cfg.get('kind')!r} is "
                    f"missing required 'delay_attr_uri' — every source must "
                    f"name a delay-poly attribute to subscribe to, there is "
                    f"no default delay (see module docstring)."
                )
            cfg["delay_feed"] = self._make_delay_feed(attr_uri)
            source_cfgs.append(cfg)

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
    if not TANGO_AVAILABLE:
        raise SystemExit("pytango is required to run this as a device server")
    run((StationSimulatorDevice,))
