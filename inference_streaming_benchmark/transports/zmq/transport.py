from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import zmq

from inference_streaming_benchmark.engine import decode_jpeg_bytes
from inference_streaming_benchmark.logging import logger

from ..base import Handler, Transport


class ZMQTransport(Transport):
    name = "zmq"
    display_name = "ZeroMQ REQ/REP (JPEG)"
    default_port = 5555

    def __init__(self):
        # server state
        self._server_ctx: zmq.Context | None = None
        self._server_socket: zmq.Socket | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # client state
        self._client_ctx: zmq.Context | None = None
        self._client_socket: zmq.Socket | None = None

    # ----- server role -----

    def start(self, port: int, handler: Handler) -> None:
        self._stop_event.clear()
        self._server_ctx = zmq.Context()
        self._server_socket = self._server_ctx.socket(zmq.REP)
        self._server_socket.bind(f"tcp://*:{port}")
        # Short poll window so we can observe stop_event promptly.
        self._server_socket.RCVTIMEO = 100

        def _serve():
            logger.info(f"zmq transport listening on tcp://*:{port}")
            while not self._stop_event.is_set():
                try:
                    image_data = self._server_socket.recv()
                except zmq.Again:
                    continue
                t0 = time.perf_counter()
                image = decode_jpeg_bytes(image_data)
                decode_ms = (time.perf_counter() - t0) * 1000
                detections, timings = handler(image)
                timings["decode_ms"] = decode_ms
                self._server_socket.send_json({"batched_detections": detections, "timings": timings})

        self._listener_thread = threading.Thread(target=_serve, daemon=True)
        self._listener_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5)
        if self._server_socket is not None:
            self._server_socket.close(linger=0)
        if self._server_ctx is not None:
            self._server_ctx.term()
        self._listener_thread = None
        self._server_socket = None
        self._server_ctx = None

    # ----- client role -----

    def connect(self, host: str, port: int) -> None:
        self._client_ctx = zmq.Context()
        self._client_socket = self._client_ctx.socket(zmq.REQ)
        self._client_socket.connect(f"tcp://{host}:{port}")

    def send(self, frame: np.ndarray):
        assert self._client_socket is not None, "connect() first"
        timings: dict[str, float] = {}
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        _, buffer = cv2.imencode(".jpg", frame)
        timings["encode_ms"] = (time.perf_counter() - t0) * 1000

        self._client_socket.send(buffer)
        response = self._client_socket.recv_json()
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        timings.update(response.get("timings", {}))
        return response["batched_detections"][0], timings

    def disconnect(self) -> None:
        if self._client_socket is not None:
            self._client_socket.close(linger=0)
        if self._client_ctx is not None:
            self._client_ctx.term()
        self._client_socket = None
        self._client_ctx = None
