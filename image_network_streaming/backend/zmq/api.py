import time

import cv2
import zmq

from image_network_streaming.logging import logger
from image_network_streaming.backend.api import BackendInterface


class ZMQBackendInterface(BackendInterface):

    def __init__(self):
        self.server_url = "tcp://localhost:5555"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server_url)

    def send_frame_to_ai_server(self, frame):
        """Send a frame to the AI server and return the response."""

        t0 = time.time()

        _, buffer = cv2.imencode(".jpg", frame)

        t1 = time.time()
        self.socket.send(buffer)

        t2 = time.time()

        # Wait for the reply from the server
        response_data = self.socket.recv_json()

        detection_results_batched = response_data["batched_detections"]
        detection_results_single = detection_results_batched[
            0
        ]  # batch size is always 1

        t3 = time.time()

        logger.info(f"send_frame_to_server total time: {t3 - t0}")
        logger.info(f"encoding time: {t1 - t0}")
        logger.info(f"send time: {t2 - t1}")
        logger.info(f"receive time: {t3 - t2}")

        return detection_results_single

    def close(self):
        self.socket.close()
        self.context.term()
