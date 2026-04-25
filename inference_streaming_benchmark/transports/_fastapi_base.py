from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI

from .base import Handler, Transport


class FastAPITransport(Transport):
    """Shared uvicorn lifecycle for FastAPI-backed transports. Subclasses provide build_app()."""

    def __init__(self):
        self._uvicorn_server: uvicorn.Server | None = None
        self._listener_thread: threading.Thread | None = None

    @classmethod
    def build_app(cls, handler: Handler) -> FastAPI:
        raise NotImplementedError

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
