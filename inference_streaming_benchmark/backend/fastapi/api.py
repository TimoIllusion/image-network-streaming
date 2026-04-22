import io
import time

import cv2
import requests

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.logging import logger


class FastAPIBackendInterface(BackendInterface):
    def __init__(self):
        self.server_url = "http://localhost:8008/detect/"
        # Reuse TCP connections across frames (avoids per-frame handshake/slowdown).
        self._session = requests.Session()

    def send_frame_to_ai_server(self, frame):
        try:
            t0 = time.time()

            _, encoded_image = cv2.imencode(".jpg", frame)

            encoding_time = time.time() - t0

            t1 = time.time()

            files = {"file": ("frame.jpg", io.BytesIO(encoded_image.tobytes()), "image/jpeg")}
            response = self._session.post(self.server_url, files=files, timeout=(2, 30))
            response_time = time.time() - t1

            logger.info(f"send_frame_to_server total time: {time.time() - t0}")
            logger.info(f"encoding time: {encoding_time}")
            logger.info(f"response time: {response_time}")

            if response.status_code == 200:
                response_data = response.json()

                detection_results_batched = response_data["batched_detections"]
                detection_results_single = detection_results_batched[0]  # batch size is always 1

                return detection_results_single
            else:
                return None

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to ai backend: {e}")
            return None
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout talking to ai backend: {e}")
            return None

    def close(self):
        try:
            self._session.close()
        except Exception:
            logger.exception("Failed to close FastAPI backend session")
