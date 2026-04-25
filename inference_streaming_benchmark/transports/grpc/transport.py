from __future__ import annotations

import time
from concurrent import futures

import grpc
import numpy as np

from inference_streaming_benchmark.logging import logger

from ..base import Handler, Transport
from ..codec import FRAME_SHAPE
from .ai_server_pb2 import (
    BoundingBox,
    DetectionResponse,
    DetectionResult,
    DetectionTimings,
    FrameRequest,
)
from .ai_server_pb2_grpc import (
    AiDetectionServiceServicer,
    AiDetectionServiceStub,
    add_AiDetectionServiceServicer_to_server,
)

_GRPC_OPTIONS = [
    ("grpc.max_send_message_length", 50 * 1024 * 1024),
    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
]


class _Servicer(AiDetectionServiceServicer):
    def __init__(self, handler: Handler):
        self._handler = handler

    def Detect(self, request, context):
        t0 = time.perf_counter()
        # gRPC sends raw ndarray bytes — decode is a cheap reshape, not JPEG.
        image = np.frombuffer(request.image, dtype=np.uint8).reshape(FRAME_SHAPE)
        if image.shape[-1] == 4:
            image = image[..., :3]
        t1 = time.perf_counter()

        detections_batched, infer_timings = self._handler(image)

        results_proto = []
        for detections in detections_batched:
            boxes, classes, scores = [], [], []
            for det in detections:
                b = det["box"]
                boxes.append(BoundingBox(x1=b["x1"], y1=b["y1"], x2=b["x2"], y2=b["y2"]))
                classes.append(str(det.get("name", det.get("class", ""))))
                scores.append(float(det.get("confidence", 0.0)))
            results_proto.append(DetectionResult(boxes=boxes, classes=classes, scores=scores))

        timings = DetectionTimings(
            decode_ms=(t1 - t0) * 1000,
            infer_ms=infer_timings.get("infer_ms", 0.0),
            post_ms=infer_timings.get("post_ms", 0.0),
        )
        return DetectionResponse(results=results_proto, timings=timings)


class GRPCTransport(Transport):
    name = "grpc"
    display_name = "gRPC unary (raw ndarray)"
    default_port = 50051

    def __init__(self):
        # server state
        self._server: grpc.Server | None = None
        # client state
        self._channel: grpc.Channel | None = None
        self._stub: AiDetectionServiceStub | None = None

    # ----- server role -----

    def start(self, port: int, handler: Handler) -> None:
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=_GRPC_OPTIONS)
        add_AiDetectionServiceServicer_to_server(_Servicer(handler), self._server)
        self._server.add_insecure_port(f"[::]:{port}")
        self._server.start()
        logger.info(f"grpc transport listening on :{port}")

    def stop(self) -> None:
        if self._server is not None:
            # grace period so in-flight RPCs can finish
            self._server.stop(grace=2).wait()
        self._server = None

    # ----- client role -----

    def connect(self, host: str, port: int) -> None:
        self._channel = grpc.insecure_channel(f"{host}:{port}", options=_GRPC_OPTIONS)
        self._stub = AiDetectionServiceStub(self._channel)

    def send(self, frame: np.ndarray):
        assert self._stub is not None, "connect() first"
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        frame_bytes = frame.tobytes()
        timings["encode_ms"] = (time.perf_counter() - t0) * 1000

        response = self._stub.Detect(FrameRequest(image=frame_bytes))
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000
        timings["decode_ms"] = response.timings.decode_ms
        timings["infer_ms"] = response.timings.infer_ms
        timings["post_ms"] = response.timings.post_ms

        if not response.results:
            return None, timings

        first = response.results[0]
        detections = []
        for box, conf, cl in zip(first.boxes, first.scores, first.classes, strict=False):
            detections.append(
                {
                    "box": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                    "confidence": conf,
                    "class": cl,
                    "name": cl,
                }
            )
        return detections, timings

    def disconnect(self) -> None:
        if self._channel is not None:
            self._channel.close()
        self._channel = None
        self._stub = None
