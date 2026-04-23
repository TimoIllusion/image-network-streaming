import time

import grpc

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.backend.grpc.ai_server_pb2 import FrameRequest
from inference_streaming_benchmark.backend.grpc.ai_server_pb2_grpc import (
    AiDetectionServiceStub,
)


class GRPCBackendInterface(BackendInterface):
    def __init__(self, host: str = "localhost", port: int = 50051):
        options = [
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50 MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel(f"{host}:{port}", options=options)
        self.stub = AiDetectionServiceStub(self.channel)

    def send_frame_to_ai_server(self, frame) -> tuple[list[dict] | None, dict]:
        timings = {}

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        frame_bytes = frame.tobytes()
        timings["encode_ms"] = (time.perf_counter() - t0) * 1000

        response = self.stub.Detect(FrameRequest(image=frame_bytes))
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        timings["decode_ms"] = response.timings.decode_ms
        timings["infer_ms"] = response.timings.infer_ms
        timings["post_ms"] = response.timings.post_ms

        detection_results = response.results
        if not detection_results:
            return None, timings

        detection_results_single = detection_results[0]
        if not detection_results_single:
            return None, timings

        results = []
        for box, conf, cl in zip(
            detection_results_single.boxes,
            detection_results_single.scores,
            detection_results_single.classes,
            strict=False,
        ):
            results.append(
                {
                    "box": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                    "confidence": conf,
                    "class": cl,
                    "name": cl,
                }
            )

        return results, timings

    def close(self):
        self.channel.close()
