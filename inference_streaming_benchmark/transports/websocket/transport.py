from __future__ import annotations

import json
import threading
import time

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect
from websockets.sync.server import serve as ws_serve

from inference_streaming_benchmark.logging import logger

from ..base import Handler, Transport
from ..codec import decode, encode
from ..envelope import build, unpack


class WebSocketTransport(Transport):
    name = "websocket"
    display_name = "WebSocket (sync)"
    default_port = 8009
    RAW = False

    def __init__(self):
        self._server = None  # websockets.sync.server.Server
        self._server_thread: threading.Thread | None = None
        self._ws: ClientConnection | None = None

    # ----- server role -----

    def start(self, port: int, handler: Handler) -> None:
        self._server = None
        raw = self.RAW
        log_name = self.name
        ready = threading.Event()

        def handle(ws):
            logger.info(f"{log_name} client connected")
            try:
                while True:
                    data = ws.recv()
                    t0 = time.perf_counter()
                    image = decode(data, raw=raw)
                    decode_ms = (time.perf_counter() - t0) * 1000
                    detections, timings = handler(image)
                    timings["decode_ms"] = decode_ms
                    ws.send(json.dumps(build(detections, timings)))
            except ConnectionClosed:
                logger.info(f"{log_name} client disconnected")

        def serve():
            with ws_serve(handle, "0.0.0.0", port, max_size=16 * 1024 * 1024, compression=None) as server:
                self._server = server
                logger.info(f"{log_name} transport listening on ws://0.0.0.0:{port}")
                ready.set()
                server.serve_forever()  # blocks until shutdown() is called

        self._server_thread = threading.Thread(target=serve, daemon=True)
        self._server_thread.start()
        ready.wait(timeout=10)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        if self._server_thread is not None:
            self._server_thread.join(timeout=5)
        self._server = None
        self._server_thread = None

    # ----- client role -----

    def connect(self, host: str, port: int) -> None:
        self._ws = ws_connect(f"ws://{host}:{port}/", max_size=16 * 1024 * 1024, compression=None)

    def send(self, frame: np.ndarray):
        assert self._ws is not None, "connect() first"
        timings: dict[str, float] = {}
        try:
            t_total = time.perf_counter()
            t0 = time.perf_counter()
            payload = encode(frame, raw=self.RAW)
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            self._ws.send(payload)
            response = self._ws.recv()
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000

            detections, server_timings = unpack(json.loads(response))
            timings.update(server_timings)
            return detections, timings
        except Exception as e:
            logger.error(f"{self.name} send failed: {e}")
            return None, timings

    def disconnect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                logger.exception(f"{self.name} disconnect failed")
            self._ws = None
