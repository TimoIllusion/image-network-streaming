from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_streaming_benchmark.config import CLIENT_NAME, CONTROL_BASE, CONTROL_HOST, UI_PORT
from inference_streaming_benchmark.transports import registry

# importing the transports package above triggers every transport's registration.
from inference_streaming_benchmark import transports  # noqa: F401  isort:skip

from . import control_client
from .mjpeg import _mjpeg_frames
from .processor import FrameProcessor
from .registration import Registrar, get_local_ip
from .state import BenchmarkCollector, CameraHandle, TransportSession

STATIC_DIR = Path(__file__).parent / "static"


camera = CameraHandle()
session = TransportSession()
collector = BenchmarkCollector()
processor = FrameProcessor(camera, session, collector)


def _build_heartbeat_stats() -> dict:
    _client, active_transport, infer = session.snapshot()
    return {
        "inference": infer,
        "mock_camera": camera.mode == "mock",
        "mock_delay_ms": camera.mock_delay_ms(),
        **collector.snapshot_for_heartbeat(active_transport),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    local_ip = get_local_ip(CONTROL_HOST)
    ui_url = f"http://{local_ip}:{UI_PORT}"
    registrar = Registrar(
        name=CLIENT_NAME,
        control_base=CONTROL_BASE,
        ui_url=ui_url,
        version="",
        stats_provider=_build_heartbeat_stats,
    )
    processor.start()
    registrar.start()
    try:
        yield
    finally:
        registrar.stop()
        processor.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    """Fetch transport availability from the AI server's control plane."""
    try:
        items = control_client.fetch_transports()
    except requests.RequestException:
        return {name: False for name in registry.all_transports()}
    return {item["name"]: True for item in items}


@app.get("/api/state")
def api_state():
    """Current local state — used by the per-device UI to initialize toggles."""
    _, active_transport, infer = session.snapshot()
    return {
        "name": CLIENT_NAME,
        "backend": active_transport,
        "inference": infer,
        "mock_camera": camera.mode == "mock",
        "mock_delay_ms": camera.mock_delay_ms(),
    }


class ControlBody(BaseModel):
    backend: str | None = None
    inference: bool | None = None
    mock_camera: bool | None = None
    mock_delay_ms: float | None = None


def _apply_backend(backend: str | None, inference: bool | None) -> None:
    """Switch backend (and possibly inference) — raises HTTPException on failure."""
    if backend is None:
        return
    if backend not in registry.all_transports():
        raise HTTPException(status_code=400, detail=f"unknown transport: {backend}")
    infer = bool(inference) if inference is not None else session.infer
    try:
        port = control_client.switch_transport(backend) if infer else 0
        session.set(backend, infer, port)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"control plane unreachable: {e}") from e


def _apply_inference_only(inference: bool) -> None:
    """Toggle inference. If enabling without a known backend, fall back to whatever the server is running."""
    if not inference:
        session.set_infer(False)
        return
    backend = session.active_transport
    if backend is None:
        try:
            items = control_client.fetch_transports()
        except requests.RequestException as e:
            raise HTTPException(status_code=503, detail=f"control plane unreachable: {e}") from e
        backend = next((it["name"] for it in items if it["active"]), None)
        if backend is None:
            raise HTTPException(status_code=400, detail="cannot enable inference: no transport active on server")
    try:
        port = control_client.switch_transport(backend)
        session.set(backend, True, port)
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"control plane unreachable: {e}") from e


@app.post("/api/control")
def api_control(body: ControlBody):
    if body.mock_camera is not None:
        camera.set_mode("mock" if body.mock_camera else "real")
    if body.mock_delay_ms is not None:
        camera.set_mock_delay_ms(body.mock_delay_ms)

    if body.backend is not None:
        _apply_backend(body.backend, body.inference)
    elif body.inference is not None:
        _apply_inference_only(body.inference)

    return {
        "ok": True,
        "backend": session.active_transport,
        "inference": session.infer,
        "mock_camera": camera.mode == "mock",
        "mock_delay_ms": camera.mock_delay_ms(),
    }


@app.post("/api/clear")
def api_clear():
    collector.clear()
    return {"ok": True}


@app.get("/api/stats")
def api_stats():
    return collector.build_stats_rows()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        _mjpeg_frames(processor),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
