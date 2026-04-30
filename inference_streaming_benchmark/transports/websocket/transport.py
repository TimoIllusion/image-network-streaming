from __future__ import annotations

import json
import threading
import time

import numpy as np
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as ws_connect
from websockets.sync.server import serve as ws_serve

from inference_streaming_benchmark.config import TRANSPORT_DEFAULT_PORTS
from inference_streaming_benchmark.logging import logger

from ..base import CLIENT_RESPONSE_TIMEOUT_S, Handler, InferenceRequest, Transport
from ..codec import decode, encode
from ..envelope import build, unpack


class WebSocketTransport(Transport):
    name = "websocket"
    display_name = "WebSocket (sync)"
    default_port = TRANSPORT_DEFAULT_PORTS[name]
    RAW = False

    def __init__(self):
        self._server = None  # websockets.sync.server.Server
        self._server_thread: threading.Thread | None = None
        self._ws: ClientConnection | None = None
        # Serializes send/recv on the persistent connection. websockets.sync raises
        # "cannot call recv concurrently" if two threads race; without this lock the
        # losing thread's response gets queued and consumed by a *later* call,
        # making total_ms appear ~0 while server timings are stale (infer/decode are
        # from a previous frame). The fix matches ZMQ REQ/REP's natural strictness.
        self._lock = threading.Lock()

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
                    try:
                        meta = json.loads(ws.recv())
                        data = ws.recv()
                        t0 = time.perf_counter()
                        image = decode(data, raw=raw)
                        decode_ms = (time.perf_counter() - t0) * 1000
                        detections, timings = handler(
                            InferenceRequest(
                                image=image,
                                client_name=meta.get("client_name", "unknown"),
                                request_id=meta.get("request_id", ""),
                                transport=log_name,
                                received_at=t0,
                            )
                        )
                        timings["decode_ms"] = decode_ms
                        ws.send(json.dumps(build(detections, timings)))
                    except ConnectionClosed:
                        raise
                    except Exception as e:
                        logger.exception(f"{log_name} request failed: {e}")
                        ws.send(json.dumps(build([], {"error": str(e)})))
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
        self._ws = ws_connect(
            f"ws://{host}:{port}/",
            max_size=16 * 1024 * 1024,
            compression=None,
            open_timeout=CLIENT_RESPONSE_TIMEOUT_S,
            close_timeout=CLIENT_RESPONSE_TIMEOUT_S,
        )

    def send(self, frame: np.ndarray, *, client_name: str = "unknown", request_id: str | None = None):
        timings: dict[str, float] = {}
        with self._lock:
            ws = self._ws
            if ws is None:
                return None, timings
            try:
                t_total = time.perf_counter()
                t0 = time.perf_counter()
                payload = encode(frame, raw=self.RAW)
                timings["encode_ms"] = (time.perf_counter() - t0) * 1000

                ws.send(json.dumps({"client_name": client_name, "request_id": request_id or ""}))
                ws.send(payload)
                response = ws.recv(timeout=CLIENT_RESPONSE_TIMEOUT_S)
                timings["total_ms"] = (time.perf_counter() - t_total) * 1000

                detections, server_timings = unpack(json.loads(response))
                timings.update(server_timings)
                return detections, timings
            except Exception as e:
                logger.error(f"{self.name} send failed: {e}")
                return None, timings

    def disconnect(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                logger.exception(f"{self.name} disconnect failed")
