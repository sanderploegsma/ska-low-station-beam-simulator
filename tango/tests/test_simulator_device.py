import concurrent.futures
import contextlib
import json
import random
import time

import grpc
import pytest
import tango
from assertpy import assert_that
from ska_control_model import HealthState
from ska_tango_testing.harness import TangoTestHarness
from ska_tango_testing.integration import TangoEventTracer

from ska_low_station_beam_simulator.simulator import StationBeamSimulator
from ska_low_station_beam_simulator.simulatorpb import (
    simulator_pb2,
    simulator_pb2_grpc,
)

UPDATE_INTERVAL_S = 0.1

pytestmark = pytest.mark.forked


class FakeSimulatorServicer(simulator_pb2_grpc.StationSimulatorServicer):
    def __init__(self):
        self.scan_running = False
        self.tick = 0

    def _fake_status(self):
        if not self.scan_running:
            return simulator_pb2.StatusResponse(
                drift_seconds=0.0,
                queue_depth=0,
                scan_running=False,
                tick_number=0,
            )

        self.tick += 1
        return simulator_pb2.StatusResponse(
            drift_seconds=random.uniform(-1.0, 1.0),
            queue_depth=random.randint(0, 1024),
            scan_running=self.scan_running,
            tick_number=self.tick,
        )

    def GetStatus(self, request, context):
        return self._fake_status()

    def StartScan(self, request, context):
        if self.scan_running:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Scan already running")
            return simulator_pb2.StartScanResponse(
                ok=False, message="Scan already running"
            )
        self.scan_running = True
        self.tick = 0
        return simulator_pb2.StartScanResponse(ok=True, message="Scan started")

    def StopScan(self, request, context):
        self.scan_running = False
        return simulator_pb2.StopScanResponse(ok=True)

    def WatchStatus(self, request, context):
        while True:
            yield self._fake_status()
            time.sleep(
                request.update_interval_s
            )  # Simulate some delay between status updates


@contextlib.contextmanager
def fake_simulator_server_factory():
    server = grpc.server(
        thread_pool=concurrent.futures.ThreadPoolExecutor(max_workers=4)
    )
    servicer = FakeSimulatorServicer()
    simulator_pb2_grpc.add_StationSimulatorServicer_to_server(servicer, server)
    port = server.add_insecure_port("localhost:0")
    server.start()
    try:
        yield f"localhost:{port}", servicer
    finally:
        server.stop(0)


@pytest.fixture
def simulator_device():
    harness = TangoTestHarness()
    harness.add_context_manager("simulator_server", fake_simulator_server_factory())
    harness.add_device(
        "test/simulator/1",
        StationBeamSimulator,
        grpc_target=lambda ctx: ctx["simulator_server"],
        status_update_interval_s=UPDATE_INTERVAL_S,
    )
    with harness as context:
        yield context.get_device("test/simulator/1")


def test_device_init(simulator_device: tango.DeviceProxy):
    assert simulator_device.healthState == HealthState.OK


def test_scan_running_attribute_reflects_simulator_state(
    simulator_device: tango.DeviceProxy,
    tracer: TangoEventTracer,
):
    # Initially, the scan should not be running
    assert simulator_device.scan_running is False
    tracer.subscribe_event(simulator_device, "scan_running")

    # Start a scan and check that the attribute reflects this
    simulator_device.StartScan(
        json.dumps(
            {
                "obs_time_epoch_s": time.time(),
                "scan_duration_s": 10,
                "scan_id": 1,
                "subarray_id": 1,
                "beam_id": 1,
            }
        )
    )
    assert_that(tracer).within_timeout(2 * UPDATE_INTERVAL_S).has_change_event_occurred(
        simulator_device,
        attribute_name="scan_running",
        attribute_value=True,
        previous_value=False,
    )

    # Stop the scan and check that the attribute reflects this
    simulator_device.StopScan()
    assert_that(tracer).within_timeout(2 * UPDATE_INTERVAL_S).has_change_event_occurred(
        simulator_device,
        attribute_name="scan_running",
        attribute_value=False,
        previous_value=True,
    )


def test_status_attributes_update_while_scanning(
    simulator_device: tango.DeviceProxy,
    tracer: TangoEventTracer,
):
    tracer.subscribe_event(simulator_device, "drift_seconds")
    tracer.subscribe_event(simulator_device, "queue_depth")
    tracer.subscribe_event(simulator_device, "tick_number")

    # Start a scan and check that the status attributes update
    simulator_device.StartScan(
        json.dumps(
            {
                "obs_time_epoch_s": time.time(),
                "scan_duration_s": 10,
                "scan_id": 1,
                "subarray_id": 1,
                "beam_id": 1,
            }
        )
    )

    assert_that(tracer).within_timeout(2 * UPDATE_INTERVAL_S).has_change_event_occurred(
        simulator_device,
        attribute_name="drift_seconds",
    ).has_change_event_occurred(
        simulator_device,
        attribute_name="queue_depth",
    ).has_change_event_occurred(
        simulator_device,
        attribute_name="tick_number",
    )


def test_start_scan_when_already_running(simulator_device: tango.DeviceProxy):
    # Start a scan
    simulator_device.StartScan(
        json.dumps(
            {
                "obs_time_epoch_s": time.time(),
                "scan_duration_s": 10,
                "scan_id": 1,
                "subarray_id": 1,
                "beam_id": 1,
            }
        )
    )

    # Attempt to start another scan while the first one is running
    with pytest.raises(tango.DevFailed):
        simulator_device.StartScan(
            json.dumps(
                {
                    "obs_time_epoch_s": time.time(),
                    "scan_duration_s": 10,
                    "scan_id": 2,
                    "subarray_id": 1,
                    "beam_id": 1,
                }
            )
        )
