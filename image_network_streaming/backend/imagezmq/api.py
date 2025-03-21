import time
import json

import imagezmq

from image_network_streaming.logging import logger
from image_network_streaming.backend.api import BackendInterface


class ImageZMQBackendInterface(BackendInterface):

    def __init__(self):
        self.sender = imagezmq.ImageSender(connect_to="tcp://localhost:5555")
        self.lock = False

    def send_frame_to_ai_server(self, frame):
        """Send a frame to the AI server and return the response."""

        t1 = time.time()

        self.lock = True
        response_data_bytes_str = self.sender.send_image("frame", frame)
        self.lock = False

        # logger.info(response_data_bytes_str)

        # Wait for the reply from the server
        response = json.loads(response_data_bytes_str)

        # logger.info(response)

        detection_results_batched = response["batched_detections"]
        detection_results_single = detection_results_batched[
            0
        ]  # batch size is always 1

        t3 = time.time()

        logger.info(f"send_frame_to_server total time: {t3 - t1}")

        return detection_results_single

    def close(self):
        while self.lock:
            logger.info("Waiting for lock to be released...")
            time.sleep(0.05)
        self.sender.close()
