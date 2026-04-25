from __future__ import annotations

from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_streaming_benchmark.transports import registry

# importing the transports package above triggers every transport's registration.
from inference_streaming_benchmark import transports  # noqa: F401  isort:skip

from . import control_client
from .mjpeg import _mjpeg_frames
from .state import BenchmarkCollector, CameraHandle, TransportSession

STATIC_DIR = Path(__file__).parent / "static"


camera = CameraHandle()
session = TransportSession()
collector = BenchmarkCollector()

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    """Fetch transport availability from the AI server's control plane.

    Returns a ``{name: online}`` dict shaped like the pre-refactor API so the
    existing JS doesn't need to change. "online" here means "known to the server";
    the currently active transport also has ``active: true`` in /api/transports.
    """
    try:
        items = control_client.fetch_transports()
    except requests.RequestException:
        # server unreachable → all transports shown offline
        return {name: False for name in registry.all_transports()}
    return {item["name"]: True for item in items}


class ControlBody(BaseModel):
    backend: str
    infer: bool


@app.post("/api/control")
def api_control(body: ControlBody):
    if body.backend not in registry.all_transports():
        raise HTTPException(status_code=400, detail=f"unknown transport: {body.backend}")
    try:
        if body.infer:
            port = control_client.switch_transport(body.backend)
        else:
            port = 0  # ignored when infer=False
        session.set(body.backend, body.infer, port)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"control plane unreachable: {e}") from e
    return {"ok": True, "backend": session.active_transport, "infer": session.infer}


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
        _mjpeg_frames(camera, session, collector),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
