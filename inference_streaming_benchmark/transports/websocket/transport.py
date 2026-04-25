from __future__ import annotations

import json
import time

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect

from inference_streaming_benchmark.logging import logger

from .._fastapi_base import FastAPITransport
from ..base import Handler
from ..codec import decode, encode
from ..envelope import build, unpack


class WebSocketTransport(FastAPITransport):
    name = "websocket"
    display_name = "WebSocket (FastAPI)"
    default_port = 8009
    RAW = False

    def __init__(self):
        super().__init__()
        self._ws: ClientConnection | None = None

    # ----- server role -----

    @classmethod
    def build_app(cls, handler: Handler) -> FastAPI:
        """Build the FastAPI app for this transport. Exposed for testability (TestClient)."""
        app = FastAPI()
        raw = cls.RAW
        log_name = cls.name

        def _infer(data: bytes):
            t = time.perf_counter()
            image = decode(data, raw=raw)
            decode_ms = (time.perf_counter() - t) * 1000
            results, timings = handler(image)
            timings["decode_ms"] = decode_ms
            return results, timings

        @app.websocket("/detect")
        async def detect(ws: WebSocket):
            await ws.accept()
            logger.info(f"{log_name} websocket client connected")
            try:
                while True:
                    t0 = time.perf_counter()
                    data = await ws.receive_bytes()
                    results, timings = await run_in_threadpool(_infer, data)
                    await ws.send_text(json.dumps(build(results, timings)))
                    logger.info(
                        f"{log_name} total={(time.perf_counter() - t0) * 1000:.1f}ms "
                        f"decode={timings['decode_ms']:.1f}ms infer={timings['infer_ms']:.1f}ms "
                        f"post={timings['post_ms']:.1f}ms"
                    )
            except WebSocketDisconnect:
                logger.info(f"{log_name} websocket client disconnected")

        return app

    # ----- client role -----

    def connect(self, host: str, port: int) -> None:
        # Persistent connection — reused across send() calls. Larger max_size to fit raw 1080p frames (~6 MB).
        self._ws = ws_connect(f"ws://{host}:{port}/detect", max_size=16 * 1024 * 1024)

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
