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
"""

from __future__ import annotations

import queue
import threading
from typing import Optional

from ska_low_station_beam_simulator.common import (
    ChannelHeap,
    QUEUE_MAXSIZE,
    ScanRunner,
    SpsPacketizer,
    StationConfig,
    log,
    sender_loop,
)
from ska_low_station_beam_simulator.direct_synthesis import DirectSynthesisStreamer

try:
    from tango import DevState
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
    first_channel_id = device_property(dtype=int, default_value=0)
    dest_ip = device_property(dtype=str, default_value="127.0.0.1")
    dest_port = device_property(dtype=int, default_value=8000)

    def init_device(self):
        super().init_device()
        self._send_queue: "queue.Queue[ChannelHeap]" = queue.Queue(
            maxsize=QUEUE_MAXSIZE
        )
        self._shutdown_event = threading.Event()
        self._scan_runner: Optional[ScanRunner] = None

        self._station_cfg = StationConfig(
            station_id=self.station_id,
            substation_id=self.substation_id,
            subarray_id=self.subarray_id,
            beam_id=self.beam_id,
            first_channel_id=self.first_channel_id,
        )
        self._packetizer = SpsPacketizer(
            self.dest_ip, self.dest_port, self._station_cfg
        )

        self._sender_thread = threading.Thread(
            target=sender_loop,
            args=(self._send_queue, self._packetizer, self._shutdown_event),
            daemon=True,
        )
        self._sender_thread.start()
        self.set_state(DevState.ON)

    @command(dtype_in=[float], doc_in="[obs_time_epoch_s, scan_duration_s, scan_id]")
    def StartScan(self, args):
        obs_time, scan_duration_s, scan_id = args
        if self._scan_runner is not None and self._scan_runner.thread.is_alive():
            raise RuntimeError("scan already running — call StopScan first")

        self._station_cfg.scan_id = int(scan_id)
        source_cfgs = [{"kind": "tone", "freq_hz": 150_000.0, "amplitude": 1.0}]
        noise_cfg = {"std": 0.05, "seed": self.station_id}

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
        self.set_state(DevState.ON)

    @attribute(dtype=int)
    def queue_depth(self):
        return self._send_queue.qsize()

    def delete_device(self):
        if self._scan_runner is not None:
            self._scan_runner.stop()
        self._shutdown_event.set()
        self._sender_thread.join(timeout=5.0)
        super().delete_device()


if __name__ == "__main__":
    if not TANGO_AVAILABLE:
        raise SystemExit("pytango is required to run this as a device server")
    run((StationSimulatorDevice,))
