from __future__ import annotations

import threading

import uvicorn
from fastapi import FastAPI

from inference_streaming_benchmark.logging import logger

DEFAULT_SIDECAR_PORT = 9000


def create_sidecar_app(transport: str, data_port: int) -> FastAPI:
    app = FastAPI()

    @app.get("/info")
    def info():
        return {"transport": transport, "port": data_port}

    @app.get("/health")
    def health():
        return {"ok": True}

    return app


def start_sidecar_in_thread(transport: str, data_port: int, sidecar_port: int = DEFAULT_SIDECAR_PORT) -> threading.Thread:
    """Start a small HTTP control server in a daemon thread.

    Advertises the active transport and its data-plane port to clients via GET /info.
    Returns the thread (daemon, dies with the main process).
    """
    app = create_sidecar_app(transport, data_port)

    def _run():
        uvicorn.run(app, host="0.0.0.0", port=sidecar_port, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    logger.info(f"Sidecar listening on :{sidecar_port} → {transport} on :{data_port}")
    return thread
