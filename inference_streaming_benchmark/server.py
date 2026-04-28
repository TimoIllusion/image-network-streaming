from __future__ import annotations

import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_streaming_benchmark.batcher import Batcher
from inference_streaming_benchmark.client_registry import ClientRegistry
from inference_streaming_benchmark.config import BATCH_ENABLED, BATCH_SIZE, BATCH_WAIT_MS, CONTROL_PORT
from inference_streaming_benchmark.engine import InferenceEngine
from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports import registry
from inference_streaming_benchmark.transports.base import Transport

LISTEN_READY_TIMEOUT_S = 5.0
PROXY_TIMEOUT_S = 5.0
SERVER_STATIC_DIR = Path(__file__).parent / "server_static"


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
        self.batcher = Batcher(
            self.engine,
            enabled=BATCH_ENABLED,
            max_batch_size=BATCH_SIZE,
            max_wait_ms=BATCH_WAIT_MS,
        )
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
                instance.start(cls.default_port, self.batcher.infer)
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
        self.batcher.stop()


class SwitchBody(BaseModel):
    name: str
    cascade: bool = False


class RegisterBody(BaseModel):
    name: str
    ui_url: str
    version: str = ""


class HeartbeatBody(BaseModel):
    name: str
    stats: dict


class ClientControlBody(BaseModel):
    backend: str | None = None
    inference: bool | None = None
    mock_camera: bool | None = None
    mock_delay_ms: float | None = None


class BatchingBody(BaseModel):
    enabled: bool | None = None
    max_batch_size: int | None = None
    max_wait_ms: float | None = None


def _cascade_to_clients(client_registry: ClientRegistry, transport_name: str) -> None:
    """After a switch, tell every registered client to reconnect to the new transport.

    Best-effort: failures are logged, never raised — one offline client should not
    block the operator's switch.
    """
    for record in client_registry.list_active():
        wants_inference = bool(record.stats.get("inference", True))
        try:
            requests.post(
                f"{record.ui_url}/api/control",
                json={"backend": transport_name, "inference": wants_inference},
                timeout=PROXY_TIMEOUT_S,
            ).raise_for_status()
            logger.info(f"cascaded transport={transport_name} to client {record.name}")
        except requests.RequestException as e:
            logger.warning(f"cascade to client {record.name} ({record.ui_url}) failed: {e}")


def build_control_app(server: Server, client_registry: ClientRegistry | None = None) -> FastAPI:
    if client_registry is None:
        client_registry = ClientRegistry()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        server.stop()

    app = FastAPI(lifespan=lifespan)

    if SERVER_STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=SERVER_STATIC_DIR), name="server-static")

    @app.get("/")
    def index():
        index_html = SERVER_STATIC_DIR / "index.html"
        if not index_html.exists():
            return {"ok": True, "message": "central UI not built"}
        return FileResponse(index_html)

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
        if body.cascade:
            _cascade_to_clients(client_registry, body.name)
        return {"name": instance.name, "port": type(instance).default_port, "active": True}

    @app.post("/register")
    def register(body: RegisterBody):
        record = client_registry.register(body.name, body.ui_url, body.version)
        logger.info(f"client registered: {record.name} @ {record.ui_url}")
        return {"ok": True, "name": record.name, "registered_at": record.registered_at}

    @app.post("/heartbeat")
    def heartbeat(body: HeartbeatBody):
        try:
            record = client_registry.heartbeat(body.name, body.stats)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"unknown client: {body.name}") from e
        return {"ok": True, "last_heartbeat_at": record.last_heartbeat_at}

    @app.get("/clients")
    def clients():
        active_name = server.active.name if server.active is not None else None
        return {
            "active_transport": active_name,
            "clients": [r.to_dict() for r in client_registry.list_active()],
        }

    @app.post("/clients/{name}/control")
    def client_control(name: str, body: ClientControlBody):
        record = client_registry.get(name)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown client: {name}")
        payload = {k: v for k, v in body.model_dump().items() if v is not None}
        if not payload:
            raise HTTPException(status_code=400, detail="control body is empty")
        try:
            r = requests.post(f"{record.ui_url}/api/control", json=payload, timeout=PROXY_TIMEOUT_S)
            r.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"client {name} unreachable: {e}") from e
        return r.json()

    @app.post("/clients/control-all")
    def clients_control_all(body: ClientControlBody):
        payload = {k: v for k, v in body.model_dump().items() if v is not None}
        if not payload:
            raise HTTPException(status_code=400, detail="control body is empty")
        backend = payload.get("backend")
        if backend is not None:
            if backend not in registry.all_transports():
                raise HTTPException(status_code=400, detail=f"unknown transport: {backend}")
            if payload.get("inference") is True:
                try:
                    server.switch(backend)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e)) from e

        records = client_registry.list_active()
        results = {}

        def post_control(record):
            try:
                r = requests.post(f"{record.ui_url}/api/control", json=payload, timeout=PROXY_TIMEOUT_S)
                r.raise_for_status()
                return record.name, "ok"
            except requests.RequestException as e:
                logger.warning(f"control-all: {record.name} ({record.ui_url}) failed: {e}")
                return record.name, "failed"

        if records:
            with ThreadPoolExecutor(max_workers=len(records)) as executor:
                futures = [executor.submit(post_control, record) for record in records]
                for future in as_completed(futures):
                    name, status = future.result()
                    results[name] = status

        return {"results": results}

    @app.post("/clients/{name}/clear")
    def client_clear(name: str):
        record = client_registry.get(name)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown client: {name}")
        try:
            r = requests.post(f"{record.ui_url}/api/clear", timeout=PROXY_TIMEOUT_S)
            r.raise_for_status()
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"client {name} unreachable: {e}") from e
        return r.json()

    @app.get("/batching")
    def batching_state():
        return server.batcher.state()

    @app.post("/batching")
    def batching_set(body: BatchingBody):
        try:
            return server.batcher.configure(
                enabled=body.enabled,
                max_batch_size=body.max_batch_size,
                max_wait_ms=body.max_wait_ms,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @app.post("/clients/clear-all")
    def clients_clear_all():
        results = {}
        for record in client_registry.list_active():
            try:
                r = requests.post(f"{record.ui_url}/api/clear", timeout=PROXY_TIMEOUT_S)
                r.raise_for_status()
                results[record.name] = "ok"
            except requests.RequestException as e:
                logger.warning(f"clear-all: {record.name} ({record.ui_url}) failed: {e}")
                results[record.name] = "failed"
        return {"results": results}

    return app


def run(default: str | None = "http_multipart") -> None:
    """Start the AI server and its control plane. Blocks until Ctrl+C."""
    server = Server()
    if default is not None:
        server.switch(default)

    app = build_control_app(server)

    logger.info(f"control plane on http://0.0.0.0:{CONTROL_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=CONTROL_PORT, log_level="warning")
