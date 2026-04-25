from __future__ import annotations

from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_streaming_benchmark.config import CONTROL_BASE, CONTROL_TIMEOUT_S
from inference_streaming_benchmark.transports import registry

# importing the transports package above triggers every transport's registration.
from inference_streaming_benchmark import transports  # noqa: F401  isort:skip

from .mjpeg import _mjpeg_frames
from .state import FrontendState

STATIC_DIR = Path(__file__).parent / "static"


state = FrontendState()
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
        r = requests.get(f"{CONTROL_BASE}/transports", timeout=CONTROL_TIMEOUT_S)
        r.raise_for_status()
    except requests.RequestException:
        # server unreachable → all transports shown offline
        return {name: False for name in registry.all_transports()}
    return {item["name"]: True for item in r.json()}


class ControlBody(BaseModel):
    backend: str
    infer: bool


@app.post("/api/control")
def api_control(body: ControlBody):
    if body.backend not in registry.all_transports():
        raise HTTPException(status_code=400, detail=f"unknown transport: {body.backend}")
    try:
        state.set_control(body.backend, body.infer)
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"control plane unreachable: {e}") from e
    return {"ok": True, "backend": state.active_transport, "infer": state.infer}


@app.post("/api/clear")
def api_clear():
    state.bench_results = {}
    return {"ok": True}


@app.get("/api/stats")
def api_stats():
    return state.build_stats_rows()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg_frames(state), media_type="multipart/x-mixed-replace; boundary=frame")
