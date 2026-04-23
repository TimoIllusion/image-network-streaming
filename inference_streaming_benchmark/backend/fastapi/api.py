import io
import time

import cv2
import requests

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.logging import logger


class FastAPIBackendInterface(BackendInterface):
    def __init__(self, host: str = "localhost", port: int = 8008):
        self.server_url = f"http://{host}:{port}/detect/"
        # Reuse TCP connections across frames (avoids per-frame handshake/slowdown).
        self._session = requests.Session()

    def send_frame_to_ai_server(self, frame):
        timings = {}
        try:
            t_total = time.perf_counter()

            t0 = time.perf_counter()
            _, encoded_image = cv2.imencode(".jpg", frame)
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            files = {"file": ("frame.jpg", io.BytesIO(encoded_image.tobytes()), "image/jpeg")}
            response = self._session.post(self.server_url, files=files, timeout=(2, 30))
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000

            if response.status_code == 200:
                data = response.json()
                timings.update(data.get("timings", {}))
                detections_batched = data["batched_detections"]
                return detections_batched[0], timings  # batch size always 1

            return None, timings

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to ai backend: {e}")
            return None, timings
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout talking to ai backend: {e}")
            return None, timings

    def close(self):
        try:
            self._session.close()
        except Exception:
            logger.exception("Failed to close FastAPI backend session")
