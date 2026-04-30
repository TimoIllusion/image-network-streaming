from __future__ import annotations

import io
import time

import numpy as np
import requests
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from inference_streaming_benchmark.config import TRANSPORT_DEFAULT_PORTS
from inference_streaming_benchmark.logging import logger

from .._fastapi_base import FastAPITransport
from ..base import CLIENT_RESPONSE_TIMEOUT_S, Handler, InferenceRequest
from ..codec import decode, encode
from ..envelope import build, unpack


class HTTPMultipartTransport(FastAPITransport):
    name = "http_multipart"
    display_name = "HTTP multipart (FastAPI)"
    default_port = TRANSPORT_DEFAULT_PORTS[name]
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

        def _infer(data: bytes, client_name: str, request_id: str):
            t = time.perf_counter()
            image = decode(data, raw=raw)
            decode_ms = (time.perf_counter() - t) * 1000
            results, timings = handler(
                InferenceRequest(
                    image=image,
                    client_name=client_name,
                    request_id=request_id,
                    transport=log_name,
                    received_at=t,
                )
            )
            timings["decode_ms"] = decode_ms
            return results, timings

        if raw:

            async def detect(request: Request):
                t0 = time.perf_counter()
                contents = await request.body()
                client_name = request.headers.get("x-infsb-client", "unknown")
                request_id = request.headers.get("x-infsb-request-id", "")
                results, timings = await run_in_threadpool(_infer, contents, client_name, request_id)
                logger.debug(
                    f"{log_name} total={(time.perf_counter() - t0) * 1000:.1f}ms "
                    f"decode={timings['decode_ms']:.1f}ms infer={timings['infer_ms']:.1f}ms "
                    f"post={timings['post_ms']:.1f}ms"
                )
                return JSONResponse(content=build(results, timings))
        else:

            async def detect(request: Request, file: UploadFile = File(...)):
                t0 = time.perf_counter()
                contents = await file.read()
                client_name = request.headers.get("x-infsb-client", "unknown")
                request_id = request.headers.get("x-infsb-request-id", "")
                results, timings = await run_in_threadpool(_infer, contents, client_name, request_id)
                logger.debug(
                    f"{log_name} total={(time.perf_counter() - t0) * 1000:.1f}ms "
                    f"decode={timings['decode_ms']:.1f}ms infer={timings['infer_ms']:.1f}ms "
                    f"post={timings['post_ms']:.1f}ms"
                )
                return JSONResponse(content=build(results, timings))

        app.post("/detect/")(detect)
        return app

    # ----- client role -----

    def connect(self, host: str, port: int) -> None:
        self._url = f"http://{host}:{port}/detect/"
        self._session = requests.Session()

    def send(self, frame: np.ndarray, *, client_name: str = "unknown", request_id: str | None = None):
        # Capture locally so a concurrent disconnect() (clearing self._session) can't
        # turn this into an AttributeError mid-request — it just becomes a clean error.
        session = self._session
        url = self._url
        timings: dict[str, float] = {}
        if session is None or url is None:
            return None, timings
        try:
            t_total = time.perf_counter()
            t0 = time.perf_counter()
            payload = encode(frame, raw=self.RAW)
            timings["encode_ms"] = (time.perf_counter() - t0) * 1000

            if self.RAW:
                response = session.post(
                    url,
                    data=payload,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-INFSB-Client": client_name,
                        "X-INFSB-Request-ID": request_id or "",
                    },
                    timeout=(2, CLIENT_RESPONSE_TIMEOUT_S),
                )
            else:
                files = {"file": ("frame.jpg", io.BytesIO(payload), "image/jpeg")}
                response = session.post(
                    url,
                    files=files,
                    headers={"X-INFSB-Client": client_name, "X-INFSB-Request-ID": request_id or ""},
                    timeout=(2, CLIENT_RESPONSE_TIMEOUT_S),
                )
            timings["total_ms"] = (time.perf_counter() - t_total) * 1000

            if response.status_code == 200:
                detections, server_timings = unpack(response.json())
                timings.update(server_timings)
                return detections, timings
            return None, timings
        except Exception as e:
            logger.error(f"{self.name} send failed: {e}")
            return None, timings

    def disconnect(self) -> None:
        if self._session is not None:
            self._session.close()
            self._session = None
        self._url = None
