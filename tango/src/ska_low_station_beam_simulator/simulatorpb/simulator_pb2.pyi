from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class DelayPolynomial(_message.Message):
    __slots__ = ("start_validity_sec", "station_id", "validity_period_sec", "xypol_coeffs_ns", "ypol_offset_ns")
    STATION_ID_FIELD_NUMBER: _ClassVar[int]
    START_VALIDITY_SEC_FIELD_NUMBER: _ClassVar[int]
    VALIDITY_PERIOD_SEC_FIELD_NUMBER: _ClassVar[int]
    XYPOL_COEFFS_NS_FIELD_NUMBER: _ClassVar[int]
    YPOL_OFFSET_NS_FIELD_NUMBER: _ClassVar[int]
    station_id: int
    start_validity_sec: float
    validity_period_sec: float
    xypol_coeffs_ns: _containers.RepeatedScalarFieldContainer[float]
    ypol_offset_ns: float
    def __init__(self, station_id: int | None = ..., start_validity_sec: float | None = ..., validity_period_sec: float | None = ..., xypol_coeffs_ns: _Iterable[float] | None = ..., ypol_offset_ns: float | None = ...) -> None: ...

class ToneSourceConfig(_message.Message):
    __slots__ = ("amplitude", "freq_hz", "source_id")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    FREQ_HZ_FIELD_NUMBER: _ClassVar[int]
    AMPLITUDE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    freq_hz: float
    amplitude: float
    def __init__(self, source_id: str | None = ..., freq_hz: float | None = ..., amplitude: float | None = ...) -> None: ...

class NoiseConfig(_message.Message):
    __slots__ = ("seed", "std")
    STD_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    std: float
    seed: int
    def __init__(self, std: float | None = ..., seed: int | None = ...) -> None: ...

class StartScanRequest(_message.Message):
    __slots__ = ("beam_id", "noise", "num_channels", "obs_time_epoch_s", "scan_duration_s", "scan_id", "subarray_id", "tone_sources")
    OBS_TIME_EPOCH_S_FIELD_NUMBER: _ClassVar[int]
    SCAN_DURATION_S_FIELD_NUMBER: _ClassVar[int]
    SCAN_ID_FIELD_NUMBER: _ClassVar[int]
    SUBARRAY_ID_FIELD_NUMBER: _ClassVar[int]
    BEAM_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    TONE_SOURCES_FIELD_NUMBER: _ClassVar[int]
    NOISE_FIELD_NUMBER: _ClassVar[int]
    obs_time_epoch_s: float
    scan_duration_s: float
    scan_id: int
    subarray_id: int
    beam_id: int
    num_channels: int
    tone_sources: _containers.RepeatedCompositeFieldContainer[ToneSourceConfig]
    noise: NoiseConfig
    def __init__(self, obs_time_epoch_s: float | None = ..., scan_duration_s: float | None = ..., scan_id: int | None = ..., subarray_id: int | None = ..., beam_id: int | None = ..., num_channels: int | None = ..., tone_sources: _Iterable[ToneSourceConfig | _Mapping] | None = ..., noise: NoiseConfig | _Mapping | None = ...) -> None: ...

class StartScanResponse(_message.Message):
    __slots__ = ("message", "ok")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    def __init__(self, ok: bool | None = ..., message: str | None = ...) -> None: ...

class StopScanRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StopScanResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool | None = ...) -> None: ...

class PushDelayUpdateRequest(_message.Message):
    __slots__ = ("polynomial", "source_id")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    POLYNOMIAL_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    polynomial: DelayPolynomial
    def __init__(self, source_id: str | None = ..., polynomial: DelayPolynomial | _Mapping | None = ...) -> None: ...

class PushDelayUpdateResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: bool | None = ...) -> None: ...

class GetStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("drift_seconds", "queue_depth", "scan_running", "tick_number")
    SCAN_RUNNING_FIELD_NUMBER: _ClassVar[int]
    QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    DRIFT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TICK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    scan_running: bool
    queue_depth: int
    drift_seconds: float
    tick_number: int
    def __init__(self, scan_running: bool | None = ..., queue_depth: int | None = ..., drift_seconds: float | None = ..., tick_number: int | None = ...) -> None: ...
