import json
import time

import imagezmq

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.logging import logger


class ImageZMQBackendInterface(BackendInterface):
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.sender = imagezmq.ImageSender(connect_to=f"tcp://{host}:{port}")
        self.lock = False

    def send_frame_to_ai_server(self, frame):
        timings = {}

        t_total = time.perf_counter()
        # ImageZMQ handles serialization internally; there's no client-side JPEG encode.
        timings["encode_ms"] = 0.0

        self.lock = True
        response_bytes = self.sender.send_image("frame", frame)
        self.lock = False

        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        response = json.loads(response_bytes)
        timings.update(response.get("timings", {}))
        detections_batched = response["batched_detections"]
        return detections_batched[0], timings

    def close(self):
        while self.lock:
            logger.info("Waiting for lock to be released...")
            time.sleep(0.05)
        self.sender.close()
