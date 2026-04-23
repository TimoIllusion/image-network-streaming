import time

import cv2
import zmq

from inference_streaming_benchmark.backend.api import BackendInterface


class ZMQBackendInterface(BackendInterface):
    def __init__(self, host: str = "localhost", port: int = 5555):
        self.server_url = f"tcp://{host}:{port}"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.server_url)

    def send_frame_to_ai_server(self, frame):
        timings = {}

        t_total = time.perf_counter()

        t0 = time.perf_counter()
        _, buffer = cv2.imencode(".jpg", frame)
        timings["encode_ms"] = (time.perf_counter() - t0) * 1000

        self.socket.send(buffer)
        response_data = self.socket.recv_json()
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        timings.update(response_data.get("timings", {}))
        detections_batched = response_data["batched_detections"]
        return detections_batched[0], timings

    def close(self):
        self.socket.close()
        self.context.term()
