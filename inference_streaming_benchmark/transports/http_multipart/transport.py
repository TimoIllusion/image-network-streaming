from __future__ import annotations

import io
import time

import numpy as np
import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from inference_streaming_benchmark.logging import logger

from .._fastapi_base import FastAPITransport
from ..base import Handler
from ..codec import decode, encode


class HTTPMultipartTransport(FastAPITransport):
    name = "http_multipart"
    display_name = "HTTP multipart (FastAPI)"
    default_port = 8008
    RAW = False

    def __init__(self):
        super().__init__()
        self._session: requests.Session | None = None
        self._url: str | None = None

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

        if raw:

            async def detect(request: Request):
                t0 = time.perf_counter()
                contents = await request.body()
                results, timings = await run_in_threadpool(_infer, contents)
                logger.info(
                    f"{log_name} total={(time.perf_counter() - t0) * 1000:.1f}ms "
                    f"decode={timings['decode_ms']:.1f}ms infer={timings['infer_ms']:.1f}ms "
                    f"post={timings['post_ms']:.1f}ms"
                )
                return JSONResponse(content={"batched_detections": results, "timings": timings})
        else:

            async def detect(file: UploadFile = File(...)):
                t0 = time.perf_counter()
                contents = await file.read()
                results, timings = await run_in_threadpool(_infer, contents)
                logger.info(
                    f"{log_name} total={(time.perf_counter() - t0) * 1000:.1f}ms "
                    f"decode={timings['decode_ms']:.1f}ms infer={timings['infer_ms']:.1f}ms "
                    f"post={timings['post_ms']:.1f}ms"
                )
                return JSONResponse(content={"batched_detections": results, "timings": timings})

        app.post("/detect/")(detect)
        return app

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
            payload = encode(frame, raw=self.RAW)
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            if self.RAW:
                response = self._session.post(
                    self._url,
                    data=payload,
                    headers={"Content-Type": "application/octet-stream"},
                    timeout=(2, 30),
                )
            else:
                files = {"file": ("frame.jpg", io.BytesIO(payload), "image/jpeg")}
                response = self._session.post(self._url, files=files, timeout=(2, 30))
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000

            if response.status_code == 200:
                data = response.json()
                timings.update(data.get("timings", {}))
                return data["batched_detections"][0], timings
            return None, timings
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.error(f"{self.name} send failed: {e}")
            return None, timings

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._url = None
