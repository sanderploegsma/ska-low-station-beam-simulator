from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DelayPolynomial(_message.Message):
    __slots__ = ("station_id", "start_validity_sec", "validity_period_sec", "xypol_coeffs_ns", "ypol_offset_ns")
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
    def __init__(self, station_id: _Optional[int] = ..., start_validity_sec: _Optional[float] = ..., validity_period_sec: _Optional[float] = ..., xypol_coeffs_ns: _Optional[_Iterable[float]] = ..., ypol_offset_ns: _Optional[float] = ...) -> None: ...

class ToneSourceConfig(_message.Message):
    __slots__ = ("source_id", "freq_hz", "amplitude")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    FREQ_HZ_FIELD_NUMBER: _ClassVar[int]
    AMPLITUDE_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    freq_hz: float
    amplitude: float
    def __init__(self, source_id: _Optional[str] = ..., freq_hz: _Optional[float] = ..., amplitude: _Optional[float] = ...) -> None: ...

class NoiseConfig(_message.Message):
    __slots__ = ("std", "seed")
    STD_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    std: float
    seed: int
    def __init__(self, std: _Optional[float] = ..., seed: _Optional[int] = ...) -> None: ...

class StartScanRequest(_message.Message):
    __slots__ = ("obs_time_epoch_s", "scan_duration_s", "scan_id", "subarray_id", "beam_id", "num_channels", "tone_sources", "noise")
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
    def __init__(self, obs_time_epoch_s: _Optional[float] = ..., scan_duration_s: _Optional[float] = ..., scan_id: _Optional[int] = ..., subarray_id: _Optional[int] = ..., beam_id: _Optional[int] = ..., num_channels: _Optional[int] = ..., tone_sources: _Optional[_Iterable[_Union[ToneSourceConfig, _Mapping]]] = ..., noise: _Optional[_Union[NoiseConfig, _Mapping]] = ...) -> None: ...

class StartScanResponse(_message.Message):
    __slots__ = ("ok", "message")
    OK_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    message: str
    def __init__(self, ok: _Optional[bool] = ..., message: _Optional[str] = ...) -> None: ...

class StopScanRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StopScanResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: _Optional[bool] = ...) -> None: ...

class PushDelayUpdateRequest(_message.Message):
    __slots__ = ("source_id", "polynomial")
    SOURCE_ID_FIELD_NUMBER: _ClassVar[int]
    POLYNOMIAL_FIELD_NUMBER: _ClassVar[int]
    source_id: str
    polynomial: DelayPolynomial
    def __init__(self, source_id: _Optional[str] = ..., polynomial: _Optional[_Union[DelayPolynomial, _Mapping]] = ...) -> None: ...

class PushDelayUpdateResponse(_message.Message):
    __slots__ = ("ok",)
    OK_FIELD_NUMBER: _ClassVar[int]
    ok: bool
    def __init__(self, ok: _Optional[bool] = ...) -> None: ...

class GetStatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class WatchStatusRequest(_message.Message):
    __slots__ = ("update_interval_s",)
    UPDATE_INTERVAL_S_FIELD_NUMBER: _ClassVar[int]
    update_interval_s: float
    def __init__(self, update_interval_s: _Optional[float] = ...) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("scan_running", "queue_depth", "drift_seconds", "tick_number")
    SCAN_RUNNING_FIELD_NUMBER: _ClassVar[int]
    QUEUE_DEPTH_FIELD_NUMBER: _ClassVar[int]
    DRIFT_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TICK_NUMBER_FIELD_NUMBER: _ClassVar[int]
    scan_running: bool
    queue_depth: int
    drift_seconds: float
    tick_number: int
    def __init__(self, scan_running: _Optional[bool] = ..., queue_depth: _Optional[int] = ..., drift_seconds: _Optional[float] = ..., tick_number: _Optional[int] = ...) -> None: ...
