import time
from concurrent import futures

import grpc
import numpy as np

from inference_streaming_benchmark.backend.grpc.ai_server_pb2 import (
    BoundingBox,
    DetectionResponse,
    DetectionResult,
    DetectionTimings,
)
from inference_streaming_benchmark.backend.grpc.ai_server_pb2_grpc import (
    AiDetectionServiceServicer,
    add_AiDetectionServiceServicer_to_server,
)
from inference_streaming_benchmark.backend.inference import get_or_load_model
from inference_streaming_benchmark.logging import logger


class AiDetectionService(AiDetectionServiceServicer):
    def __init__(self):
        self.model = get_or_load_model()

    def Detect(self, request, context):
        t0 = time.perf_counter()

        image = np.frombuffer(request.image, dtype=np.uint8).reshape((1080, 1920, 3))
        if image.shape[-1] == 4:
            image = image[..., :3]

        t1 = time.perf_counter()
        results = self.model(image)
        t2 = time.perf_counter()

        detection_results = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            boxes_converted = []
            confs_converted = []
            classes_converted = []
            for box, conf, cl in zip(boxes, confs, classes, strict=False):
                x1, y1, x2, y2 = box
                boxes_converted.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
                classes_converted.append(str(cl))
                confs_converted.append(conf)

            detection_results.append(DetectionResult(boxes=boxes_converted, classes=classes_converted, scores=confs_converted))

        t3 = time.perf_counter()

        timings = DetectionTimings(
            decode_ms=(t1 - t0) * 1000,
            infer_ms=(t2 - t1) * 1000,
            post_ms=(t3 - t2) * 1000,
        )

        logger.info(
            "detect total=%.1fms decode=%.1fms infer=%.1fms post=%.1fms",
            (t3 - t0) * 1000,
            timings.decode_ms,
            timings.infer_ms,
            timings.post_ms,
        )

        return DetectionResponse(results=detection_results, timings=timings)


def serve():
    logger.info("Starting GRPC AI server...")
    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    add_AiDetectionServiceServicer_to_server(AiDetectionService(), server)
    port = 50051
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"GRPC AI server started! Listening on port {port}.")
    server.wait_for_termination()
