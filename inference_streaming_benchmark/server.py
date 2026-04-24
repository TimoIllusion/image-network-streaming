from __future__ import annotations

import socket
import threading
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from inference_streaming_benchmark.engine import InferenceEngine
from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports import registry
from inference_streaming_benchmark.transports.base import Transport

CONTROL_PORT = 9000
LISTEN_READY_TIMEOUT_S = 5.0


def _wait_until_listening(port: int, host: str = "localhost", timeout: float = LISTEN_READY_TIMEOUT_S) -> bool:
    """Poll `host:port` until a TCP connect succeeds. Returns True on success."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return True
        except OSError:
            time.sleep(0.05)
    return False


class Server:
    """Hosts one active transport at a time; hot-swaps on request."""

    def __init__(self):
        self.engine = InferenceEngine()
        self.active: Transport | None = None
        self._lock = threading.Lock()

    def switch(self, name: str) -> Transport:
        with self._lock:
            if self.active is not None and self.active.name == name:
                return self.active
            if self.active is not None:
                logger.info(f"stopping transport: {self.active.name}")
                self.active.stop()
                self.active = None

            cls = registry.get(name)
            instance = cls()
            logger.info(f"starting transport: {name} on port {cls.default_port}")
            try:
                instance.start(cls.default_port, self.engine.infer)
            except Exception:
                logger.exception(f"failed to start transport {name!r}")
                raise

            if not _wait_until_listening(cls.default_port):
                instance.stop()
                raise RuntimeError(f"transport {name!r} did not start listening on port {cls.default_port}")

            self.active = instance
            return instance

    def stop(self) -> None:
        with self._lock:
            if self.active is not None:
                logger.info(f"stopping transport: {self.active.name}")
                self.active.stop()
                self.active = None


class SwitchBody(BaseModel):
    name: str


def build_control_app(server: Server) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/transports")
    def transports():
        active_name = server.active.name if server.active is not None else None
        out = []
        for name, cls in registry.all_transports().items():
            out.append(
                {
                    "name": name,
                    "display_name": cls.display_name,
                    "port": cls.default_port,
                    "active": name == active_name,
                }
            )
        return out

    @app.post("/switch")
    def switch(body: SwitchBody):
        if body.name not in registry.all_transports():
            raise HTTPException(status_code=400, detail=f"unknown transport: {body.name}")
        try:
            instance = server.switch(body.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        return {"name": instance.name, "port": type(instance).default_port, "active": True}

    return app


def run(default: str | None = "http_multipart") -> None:
    """Start the AI server and its control plane. Blocks until Ctrl+C."""
    server = Server()
    if default is not None:
        server.switch(default)

    app = build_control_app(server)

    def _shutdown():
        server.stop()

    app.add_event_handler("shutdown", _shutdown)

    logger.info(f"control plane on http://0.0.0.0:{CONTROL_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=CONTROL_PORT, log_level="warning")
