from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect

from inference_streaming_benchmark.engine import decode_jpeg_bytes
from inference_streaming_benchmark.logging import logger

from ..base import Handler, Transport

# Raw payload shape — kept in sync with grpc/transport.py:38 and frontend.py:_open_camera.
_RAW_SHAPE = (1080, 1920, 3)


class WebSocketTransport(Transport):
    name = "websocket"
    display_name = "WebSocket (FastAPI)"
    default_port = 8009
    RAW = False

    def __init__(self):
        self._uvicorn_server: uvicorn.Server | None = None
        self._listener_thread: threading.Thread | None = None
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
            if raw:
                image = np.frombuffer(data, dtype=np.uint8).reshape(_RAW_SHAPE)
            else:
                image = decode_jpeg_bytes(data)
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
                    await ws.send_text(json.dumps({"batched_detections": results, "timings": timings}))
                    logger.info(
                        f"{log_name} total={(time.perf_counter() - t0) * 1000:.1f}ms "
                        f"decode={timings['decode_ms']:.1f}ms infer={timings['infer_ms']:.1f}ms "
                        f"post={timings['post_ms']:.1f}ms"
                    )
            except WebSocketDisconnect:
                logger.info(f"{log_name} websocket client disconnected")

        return app

    def start(self, port: int, handler: Handler) -> None:
        app = self.build_app(handler)
        config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)
        self._listener_thread = threading.Thread(target=self._uvicorn_server.run, daemon=True)
        self._listener_thread.start()

    def stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
        if self._listener_thread is not None:
            self._listener_thread.join(timeout=5)
        self._uvicorn_server = None
        self._listener_thread = None

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
            if self.RAW:
                payload = frame.tobytes()
            else:
                _, encoded = cv2.imencode(".jpg", frame)
                payload = encoded.tobytes()
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            self._ws.send(payload)
            response = self._ws.recv()
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000

            data = json.loads(response)
            timings.update(data.get("timings", {}))
            return data["batched_detections"][0], timings
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


class WebSocketRawTransport(WebSocketTransport):
    name = "websocket_raw"
    display_name = "WebSocket raw (FastAPI, ndarray)"
    default_port = 8011
    RAW = True
