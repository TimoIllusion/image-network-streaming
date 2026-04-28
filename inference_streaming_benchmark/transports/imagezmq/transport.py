from __future__ import annotations

import json
import threading
import time

import imagezmq
import numpy as np

from inference_streaming_benchmark.logging import logger

from ..base import Handler, InferenceRequest, Transport
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
                    name, image = self._hub.recv_image()
                except Exception:
                    # imagezmq surfaces zmq.Again as a generic exception; just retry.
                    continue
                client_name, sep, request_id = name.partition("|")
                # imagezmq delivers a numpy ndarray directly — no JPEG decode on our side.
                decode_ms = 0.0
                detections, timings = handler(
                    InferenceRequest(
                        image=image,
                        client_name=client_name or "unknown",
                        request_id=request_id if sep else "",
                        transport="imagezmq",
                        received_at=time.perf_counter(),
                    )
                )
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
        sender = imagezmq.ImageSender(connect_to=f"tcp://{host}:{port}")
        # Without a recv timeout, a server teardown mid-request leaves the REQ socket
        # blocking on recv forever — and disconnect() from another thread can't safely
        # cancel it. 5s matches the cascade window.
        sender.zmq_socket.RCVTIMEO = 5000
        self._sender = sender

    def send(self, frame: np.ndarray, *, client_name: str = "unknown", request_id: str | None = None):
        sender = self._sender
        timings: dict[str, float] = {"encode_ms": 0.0}
        if sender is None:
            return None, timings
        try:
            t_total = time.perf_counter()
            response_bytes = sender.send_image(f"{client_name}|{request_id or ''}", frame)
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000
            detections, server_timings = unpack(json.loads(response_bytes))
            timings.update(server_timings)
            return detections, timings
        except Exception as e:
            logger.error(f"{self.name} send failed: {e}")
            return None, timings

    def disconnect(self) -> None:
        sender = self._sender
        self._sender = None
        if sender is not None:
            try:
                # linger=0 cancels any in-flight recv on this socket immediately,
                # so a wedged send() in another thread unblocks instead of hanging
                # the swap.
                sender.zmq_socket.close(linger=0)
            except Exception:
                logger.exception(f"{self.name} disconnect failed")
