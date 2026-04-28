from __future__ import annotations

import time
from concurrent import futures

import grpc
import numpy as np

from inference_streaming_benchmark.logging import logger

from ..base import Handler, InferenceRequest, Transport
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
        meta = dict(context.invocation_metadata())
        t0 = time.perf_counter()
        # gRPC sends raw ndarray bytes — decode is a cheap reshape, not JPEG.
        image = np.frombuffer(request.image, dtype=np.uint8).reshape(FRAME_SHAPE)
        if image.shape[-1] == 4:
            image = image[..., :3]
        t1 = time.perf_counter()

        detections_batched, infer_timings = self._handler(
            InferenceRequest(
                image=image,
                client_name=meta.get("x-infsb-client", "unknown"),
                request_id=meta.get("x-infsb-request-id", ""),
                transport="grpc",
                received_at=t0,
            )
        )

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
        context.set_trailing_metadata(
            (
                ("x-infsb-queue-wait-ms", str(infer_timings.get("queue_wait_ms", 0.0))),
                ("x-infsb-backlog-wait-ms", str(infer_timings.get("backlog_wait_ms", 0.0))),
                ("x-infsb-batch-fill-wait-ms", str(infer_timings.get("batch_fill_wait_ms", 0.0))),
                ("x-infsb-batch-size", str(infer_timings.get("batch_size", 1))),
                ("x-infsb-batching-enabled", "1" if infer_timings.get("batching_enabled", False) else "0"),
                ("x-infsb-batching-max-batch-size", str(infer_timings.get("batching_max_batch_size", 1))),
                ("x-infsb-batching-max-wait-ms", str(infer_timings.get("batching_max_wait_ms", 0.0))),
            )
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
        channel = grpc.insecure_channel(f"{host}:{port}", options=_GRPC_OPTIONS)
        self._channel = channel
        self._stub = AiDetectionServiceStub(channel)

    def send(self, frame: np.ndarray, *, client_name: str = "unknown", request_id: str | None = None):
        stub = self._stub
        timings: dict[str, float] = {}
        if stub is None:
            return None, timings
        try:
            t_total = time.perf_counter()
            t0 = time.perf_counter()
            frame_bytes = frame.tobytes()
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            # 5s deadline matches the cascade window — without it a server teardown
            # mid-RPC could keep the call hanging until grpc's default keepalive trips.
            response, call = stub.Detect.with_call(
                FrameRequest(image=frame_bytes),
                timeout=5.0,
                metadata=(("x-infsb-client", client_name), ("x-infsb-request-id", request_id or "")),
            )
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000
            timings["decode_ms"] = response.timings.decode_ms
            timings["infer_ms"] = response.timings.infer_ms
            timings["post_ms"] = response.timings.post_ms
            trailing = dict(call.trailing_metadata() or ())
            timings["queue_wait_ms"] = float(trailing.get("x-infsb-queue-wait-ms", 0.0))
            timings["backlog_wait_ms"] = float(trailing.get("x-infsb-backlog-wait-ms", 0.0))
            timings["batch_fill_wait_ms"] = float(trailing.get("x-infsb-batch-fill-wait-ms", 0.0))
            timings["batch_size"] = float(trailing.get("x-infsb-batch-size", 1))
            timings["batching_enabled"] = trailing.get("x-infsb-batching-enabled", "0") == "1"
            timings["batching_max_batch_size"] = float(trailing.get("x-infsb-batching-max-batch-size", 1))
            timings["batching_max_wait_ms"] = float(trailing.get("x-infsb-batching-max-wait-ms", 0.0))

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
        except Exception as e:
            logger.error(f"{self.name} send failed: {e}")
            return None, timings

    def disconnect(self) -> None:
        channel = self._channel
        self._channel = None
        self._stub = None
        if channel is not None:
            try:
                channel.close()
            except Exception:
                logger.exception(f"{self.name} disconnect failed")
