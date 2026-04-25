from __future__ import annotations

import json
import threading
import time

import imagezmq
import numpy as np

from inference_streaming_benchmark.logging import logger

from ..base import Handler, Transport
from ..envelope import build, unpack


class ImageZMQTransport(Transport):
    name = "imagezmq"
    display_name = "ImageZMQ (raw ndarray)"
    default_port = 5556

    def __init__(self):
        # server state
        self._hub: imagezmq.ImageHub | None = None
        self._listener_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # client state
        self._sender: imagezmq.ImageSender | None = None
        self._sending = False

    # ----- server role -----

    def start(self, port: int, handler: Handler) -> None:
        self._stop_event.clear()
        self._hub = imagezmq.ImageHub(open_port=f"tcp://0.0.0.0:{port}")

        def _serve():
            logger.info(f"imagezmq transport listening on tcp://0.0.0.0:{port}")
            assert self._hub is not None
            # imagezmq's recv_image blocks; use a short socket timeout so we can exit on stop.
            self._hub.zmq_socket.RCVTIMEO = 100
            while not self._stop_event.is_set():
                try:
                    _, image = self._hub.recv_image()
                except Exception:
                    # imagezmq surfaces zmq.Again as a generic exception; just retry.
                    continue
                # imagezmq delivers a numpy ndarray directly — no JPEG decode on our side.
                decode_ms = 0.0
                detections, timings = handler(image)
                timings["decode_ms"] = decode_ms
                self._hub.send_reply(json.dumps(build(detections, timings)).encode())

        self._listener_thread = threading.Thread(target=_serve, daemon=True)
        self._listener_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5)
        if self._hub is not None:
            self._hub.close()
        self._listener_thread = None
        self._hub = None

    # ----- client role -----

    def connect(self, host: str, port: int) -> None:
        self._sender = imagezmq.ImageSender(connect_to=f"tcp://{host}:{port}")

    def send(self, frame: np.ndarray):
        assert self._sender is not None, "connect() first"
        timings: dict[str, float] = {}
        timings["encode_ms"] = 0.0  # no client-side encode — imagezmq sends raw ndarray

        t_total = time.perf_counter()
        self._sending = True
        response_bytes = self._sender.send_image("frame", frame)
        self._sending = False
        timings["total_ms"] = (time.perf_counter() - t_total) * 1000

        detections, server_timings = unpack(json.loads(response_bytes))
        timings.update(server_timings)
        return detections, timings

    def disconnect(self) -> None:
        while self._sending:
            time.sleep(0.05)
        if self._sender is not None:
            self._sender.close()
        self._sender = None
