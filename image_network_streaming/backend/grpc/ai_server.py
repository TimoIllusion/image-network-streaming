import time
from concurrent import futures
from typing import List

import grpc

import numpy as np

from PIL import Image
import io
from ultralytics import YOLO
from ultralytics.engine.results import Results

from image_network_streaming.backend.grpc.ai_server_pb2 import (
    DetectionResponse,
    DetectionResult,
    BoundingBox,
)
from image_network_streaming.backend.grpc.ai_server_pb2_grpc import (
    AiDetectionServiceServicer,
    add_AiDetectionServiceServicer_to_server,
)

from image_network_streaming.logging import logger


class AiDetectionService(AiDetectionServiceServicer):

    def __init__(self):
        # Load your YOLO model here
        self.model = YOLO("yolov8n.pt")

    def Detect(self, request, context):
        t0 = time.time()

        logger.info("Detect called.")

        # Convert bytes to image
        # image_bytes = io.BytesIO(request.image)
        # image = Image.open(image_bytes)
        # image = np.array(image)

        image = np.frombuffer(request.image, dtype=np.uint8).reshape((1080, 1920, 3))

        # Ensure image is in RGB format if it's a PNG with an alpha channel
        if image.shape[-1] == 4:
            image = image[..., :3]

        t1 = time.time()
        # Perform inference using YOLO
        results: List[Results] = self.model(image)
        t2 = time.time()

        # Prepare the response
        detection_results = []

        # Process results list
        for result in results:

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            boxes_converted = []
            confs_converted = []
            classes_converted = []
            for box, conf, cl in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box

                bounding_box_proto = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

                boxes_converted.append(bounding_box_proto)
                classes_converted.append(str(cl))
                confs_converted.append(conf)

            detection_result = DetectionResult(
                boxes=boxes_converted, classes=classes_converted, scores=confs_converted
            )

            detection_results.append(detection_result)

        response = DetectionResponse(results=detection_results)

        t3 = time.time()

        logger.info(f"GRPCBackendInterface total time: {t3 - t0}")
        logger.info(f"decoding time: {t1 - t0}")
        logger.info(f"inference time: {t2 - t1}")
        logger.info(f"postprocessing time: {t3 - t2}")

        return response


def serve():
    logger.info("Starting GRPC AI server...")
    options = [
        ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50 MB
        ("grpc.max_receive_message_length", 50 * 1024 * 1024),  # 50 MB
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    add_AiDetectionServiceServicer_to_server(AiDetectionService(), server)
    port = 50051
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"GRPC AI server started! Listening on port {port}.")
    server.wait_for_termination()
