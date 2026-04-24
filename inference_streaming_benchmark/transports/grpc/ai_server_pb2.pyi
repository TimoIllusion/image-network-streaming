from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FrameRequest(_message.Message):
    __slots__ = ("image",)
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    def __init__(self, image: _Optional[bytes] = ...) -> None: ...

class DetectionResult(_message.Message):
    __slots__ = ("boxes", "classes", "scores")
    BOXES_FIELD_NUMBER: _ClassVar[int]
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    boxes: _containers.RepeatedCompositeFieldContainer[BoundingBox]
    classes: _containers.RepeatedScalarFieldContainer[str]
    scores: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, boxes: _Optional[_Iterable[_Union[BoundingBox, _Mapping]]] = ..., classes: _Optional[_Iterable[str]] = ..., scores: _Optional[_Iterable[float]] = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("x1", "y1", "x2", "y2")
    X1_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    x1: float
    y1: float
    x2: float
    y2: float
    def __init__(self, x1: _Optional[float] = ..., y1: _Optional[float] = ..., x2: _Optional[float] = ..., y2: _Optional[float] = ...) -> None: ...

class DetectionTimings(_message.Message):
    __slots__ = ("decode_ms", "infer_ms", "post_ms")
    DECODE_MS_FIELD_NUMBER: _ClassVar[int]
    INFER_MS_FIELD_NUMBER: _ClassVar[int]
    POST_MS_FIELD_NUMBER: _ClassVar[int]
    decode_ms: float
    infer_ms: float
    post_ms: float
    def __init__(self, decode_ms: _Optional[float] = ..., infer_ms: _Optional[float] = ..., post_ms: _Optional[float] = ...) -> None: ...

class DetectionResponse(_message.Message):
    __slots__ = ("results", "timings")
    RESULTS_FIELD_NUMBER: _ClassVar[int]
    TIMINGS_FIELD_NUMBER: _ClassVar[int]
    results: _containers.RepeatedCompositeFieldContainer[DetectionResult]
    timings: DetectionTimings
    def __init__(self, results: _Optional[_Iterable[_Union[DetectionResult, _Mapping]]] = ..., timings: _Optional[_Union[DetectionTimings, _Mapping]] = ...) -> None: ...
