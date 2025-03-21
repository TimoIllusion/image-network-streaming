import time
from typing import List

import grpc
import cv2

from image_network_streaming.backend.api import BackendInterface

from image_network_streaming.backend.grpc.ai_server_pb2 import FrameRequest
from image_network_streaming.backend.grpc.ai_server_pb2_grpc import (
    AiDetectionServiceStub,
)

from image_network_streaming.logging import logger


class GRPCBackendInterface(BackendInterface):

    def __init__(self):
        options = [
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50 MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50 MB
        ]
        self.channel = grpc.insecure_channel("localhost:50051", options=options)
        self.stub = AiDetectionServiceStub(self.channel)

    def send_frame_to_ai_server(self, frame) -> List[dict]:
        t0 = time.time()
        # _, encoded_image = cv2.imencode(".jpg", frame)
        # frame_bytes = encoded_image.tobytes()
        frame_bytes = frame.tobytes()
        t1 = time.time()

        response = self.stub.Detect(FrameRequest(image=frame_bytes))
        # Process response
        detection_results = response.results
        # Assume batch size is always 1 for simplicity
        detection_results_single = detection_results[0]

        if detection_results_single:
            results = []
            for box, conf, cl in zip(
                detection_results_single.boxes,
                detection_results_single.scores,
                detection_results_single.classes,
            ):

                detection = {
                    "box": {"x1": box.x1, "y1": box.y1, "x2": box.x2, "y2": box.y2},
                    "confidence": conf,
                    "class": cl,
                    "name": cl,
                }

                results.append(detection)

                t2 = time.time()

                logger.info(f"GRPCBackendInterface total time: {t2 - t0}")
                logger.info(f"encoding time: {t1 - t0}")
                logger.info(f"response time: {t2 - t1}")

            return results

        else:
            return None

    def close(self):
        self.channel.close()
