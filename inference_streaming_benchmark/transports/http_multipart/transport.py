from __future__ import annotations

import io
import threading
import time

import cv2
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from inference_streaming_benchmark.engine import decode_jpeg_bytes
from inference_streaming_benchmark.logging import logger

from ..base import Handler, Transport


class HTTPMultipartTransport(Transport):
    name = "http_multipart"
    display_name = "HTTP multipart (FastAPI)"
    default_port = 8008

    def __init__(self):
        self._uvicorn_server: uvicorn.Server | None = None
        self._listener_thread: threading.Thread | None = None
        self._session: requests.Session | None = None
        self._url: str | None = None

    # ----- server role -----

    @staticmethod
    def build_app(handler: Handler) -> FastAPI:
        """Build the FastAPI app for this transport. Exposed for testability (TestClient)."""
        app = FastAPI()

        async def detect(file: UploadFile = File(...)):
            t0 = time.perf_counter()
            contents = await file.read()

            def _infer(data: bytes):
                t = time.perf_counter()
                image = decode_jpeg_bytes(data)
                decode_ms = (time.perf_counter() - t) * 1000
                results, timings = handler(image)
                timings["decode_ms"] = decode_ms
                return results, timings

            results, timings = await run_in_threadpool(_infer, contents)
            logger.info(
                "http_multipart total=%.1fms decode=%.1fms infer=%.1fms post=%.1fms",
                (time.perf_counter() - t0) * 1000,
                timings["decode_ms"],
                timings["infer_ms"],
                timings["post_ms"],
            )
            return JSONResponse(content={"batched_detections": results, "timings": timings})

        app.post("/detect/")(detect)
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
        self._url = f"http://{host}:{port}/detect/"
        self._session = requests.Session()

    def send(self, frame: np.ndarray):
        assert self._session is not None and self._url is not None, "connect() first"
        timings: dict[str, float] = {}
        try:
            t_total = time.perf_counter()
            t0 = time.perf_counter()
            _, encoded = cv2.imencode(".jpg", frame)
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            files = {"file": ("frame.jpg", io.BytesIO(encoded.tobytes()), "image/jpeg")}
            response = self._session.post(self._url, files=files, timeout=(2, 30))
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000

            if response.status_code == 200:
                data = response.json()
                timings.update(data.get("timings", {}))
                return data["batched_detections"][0], timings
            return None, timings
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.error(f"http_multipart send failed: {e}")
            return None, timings

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._url = None
