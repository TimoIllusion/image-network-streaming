import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import statistics
import threading
import time
from pathlib import Path

import cv2
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.frontend.media import draw_detections, draw_fps
from inference_streaming_benchmark.logging import logger

# Hardcoded transport registry: name → (data port, sidecar port).
# Host is assumed to be localhost for the status probe and the client.
TRANSPORTS: dict[str, dict[str, int]] = {
    "http_multipart": {"data_port": 8008, "sidecar_port": 9001},
    "zmq": {"data_port": 5555, "sidecar_port": 9002},
    "imagezmq": {"data_port": 5556, "sidecar_port": 9003},
    "grpc": {"data_port": 50051, "sidecar_port": 9004},
}

HOST = "localhost"
HEALTH_TIMEOUT_S = 0.2
UI_PORT = 8501

# Columns we collect per frame and show as medians in the stats table.
# transmission_ms = total - infer: end-to-end cost excluding only AI inference,
# useful for comparing transport protocols (includes encode/decode/comms/post).
TIMING_COLUMNS = ("encode_ms", "decode_ms", "infer_ms", "post_ms", "comms_ms", "transmission_ms", "total_ms")

STATIC_DIR = Path(__file__).parent / "inference_streaming_benchmark" / "frontend" / "static"


def _create_backend(transport: str, host: str, port: int) -> BackendInterface:
    if transport == "http_multipart":
        from inference_streaming_benchmark.backend.http_multipart.api import HTTPMultipartBackendInterface

        return HTTPMultipartBackendInterface(host=host, port=port)
    if transport == "zmq":
        from inference_streaming_benchmark.backend.zmq.api import ZMQBackendInterface

        return ZMQBackendInterface(host=host, port=port)
    if transport == "imagezmq":
        from inference_streaming_benchmark.backend.imagezmq.api import ImageZMQBackendInterface

        return ImageZMQBackendInterface(host=host, port=port)
    if transport == "grpc":
        from inference_streaming_benchmark.backend.grpc.api import GRPCBackendInterface

        return GRPCBackendInterface(host=host, port=port)
    raise ValueError(f"Unknown transport: {transport}")


def _probe_sidecar(sidecar_port: int) -> bool:
    try:
        r = requests.get(f"http://{HOST}:{sidecar_port}/health", timeout=HEALTH_TIMEOUT_S)
        return r.ok
    except requests.RequestException:
        return False


class FrontendState:
    """Single-process state shared between the MJPEG generator and the control endpoints.

    The lock guards backend swaps so the video generator never sees a half-closed interface.
    """

    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self.backend: BackendInterface | None = None
        self.active_transport: str | None = None
        self.infer: bool = False
        self.bench_results: dict = {}
        self.lock = threading.Lock()

    def ensure_camera(self):
        if self.cap is None:
            logger.info("Start camera initialization")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            self.cap = cap
            logger.info("Camera initialized")

    def set_control(self, backend: str, infer: bool):
        with self.lock:
            self.infer = infer
            if not infer:
                if self.backend is not None:
                    self.backend.close()
                    self.backend = None
                    self.active_transport = None
                    logger.info("Backend comms closed")
                return

            if self.active_transport != backend or self.backend is None:
                if self.backend is not None:
                    self.backend.close()
                    logger.info("Backend comms closed (switching to {})", backend)
                port = TRANSPORTS[backend]["data_port"]
                self.backend = _create_backend(backend, HOST, port)
                self.active_transport = backend
                logger.info(f"Backend comms initialized: {backend}")

    def record_timing(self, backend: str, timings: dict):
        # comms = pure network transit (total minus all endpoint processing)
        server_ms = timings.get("decode_ms", 0.0) + timings.get("infer_ms", 0.0) + timings.get("post_ms", 0.0)
        comms_ms = max(0.0, timings.get("total_ms", 0.0) - timings.get("encode_ms", 0.0) - server_ms)
        # transmission = everything but AI inference (encode + decode + comms + post)
        transmission_ms = max(0.0, timings.get("total_ms", 0.0) - timings.get("infer_ms", 0.0))
        timings = {**timings, "comms_ms": comms_ms, "transmission_ms": transmission_ms}

        bench = self.bench_results.setdefault(backend, {"active_time_s": 0.0, **{col: [] for col in TIMING_COLUMNS}})
        for col in TIMING_COLUMNS:
            if col in timings:
                bench[col].append(timings[col])
        bench["active_time_s"] += timings.get("total_ms", 0.0) / 1000

    def build_stats_rows(self) -> list[dict]:
        rows = []
        for backend, data in self.bench_results.items():
            totals = data["total_ms"]
            if not totals:
                continue
            duration_s = data["active_time_s"]
            row = {
                "Backend": backend,
                "Frames": len(totals),
                "Duration (s)": f"{duration_s:.1f}",
                "FPS": f"{len(totals) / duration_s:.1f}" if duration_s > 0 else "-",
            }
            for col in TIMING_COLUMNS:
                samples = data[col]
                row[col.replace("_ms", " (ms)")] = f"{statistics.median(samples):.1f}" if samples else "-"
            rows.append(row)
        return rows


state = FrontendState()
app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def api_status():
    return {name: _probe_sidecar(cfg["sidecar_port"]) for name, cfg in TRANSPORTS.items()}


class ControlBody(BaseModel):
    backend: str
    infer: bool


@app.post("/api/control")
def api_control(body: ControlBody):
    if body.backend not in TRANSPORTS:
        return JSONResponse({"error": f"unknown backend: {body.backend}"}, status_code=400)
    state.set_control(body.backend, body.infer)
    return {"ok": True, "backend": state.active_transport, "infer": state.infer}


@app.post("/api/clear")
def api_clear():
    state.bench_results = {}
    return {"ok": True}


@app.get("/api/stats")
def api_stats():
    return state.build_stats_rows()


def _mjpeg_frames():
    """Blocking generator: captures frames, runs inference when enabled, yields MJPEG parts."""
    state.ensure_camera()
    assert state.cap is not None

    while True:
        ret, frame = state.cap.read()
        if not ret:
            logger.warning("Failed to capture frame")
            time.sleep(0.01)
            continue

        # Snapshot the backend under the lock so a swap mid-send never races.
        with state.lock:
            active_backend = state.backend
            active_transport = state.active_transport
            infer = state.infer

        if infer and active_backend is not None and active_transport is not None:
            detections, timings = active_backend.send_frame_to_ai_server(frame)
            state.record_timing(active_transport, timings)
            fps = 1000 / timings["total_ms"] if timings.get("total_ms", 0) > 0 else 0
            if detections is not None:
                frame = draw_detections(frame, detections)
                frame = draw_fps(frame, fps)

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        jpg = buf.tobytes()
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=UI_PORT)
