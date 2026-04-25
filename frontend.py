import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import statistics
import threading
import time
from pathlib import Path

import cv2
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from inference_streaming_benchmark.frontend.media import draw_detections, draw_fps
from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports import registry
from inference_streaming_benchmark.transports.base import Transport
from inference_streaming_benchmark.transports.codec import FRAME_SHAPE

# importing the transports package above triggers every transport's registration.
from inference_streaming_benchmark import transports  # noqa: F401  isort:skip

HOST = "localhost"
UI_PORT = 8501
CONTROL_BASE = "http://localhost:9000"
CONTROL_TIMEOUT_S = 5.0

# Columns we collect per frame and show as medians in the stats table.
# transmission_ms = total - infer: end-to-end cost excluding only AI inference.
TIMING_COLUMNS = ("encode_ms", "decode_ms", "infer_ms", "post_ms", "comms_ms", "transmission_ms", "total_ms")

STATIC_DIR = Path(__file__).parent / "inference_streaming_benchmark" / "frontend" / "static"


_MOCK_IMAGE_PATH = Path(__file__).parent / "resources" / "example_dall_e.png"


class _FakeVideoCapture:
    """Static 1920×1080 BGR frames for Claude-driven smoke tests.

    Enabled by MOCK_CAMERA=1. Returns the resources/example_dall_e.png image
    (resized to 1920×1080) with a timestamp overlay at ~30fps. The image
    contains people, chairs, a dining table, and a laptop so YOLO produces
    real detections that exercise draw_detections.
    """

    def __init__(self):
        self._t0 = time.time()
        self._released = False
        img = cv2.imread(str(_MOCK_IMAGE_PATH))
        if img is None:
            raise RuntimeError(f"MOCK_CAMERA: could not load {_MOCK_IMAGE_PATH}")
        h, w = FRAME_SHAPE[0], FRAME_SHAPE[1]
        self._base = cv2.resize(img, (w, h))

    def read(self):
        if self._released:
            return False, None
        frame = self._base.copy()
        elapsed = time.time() - self._t0
        cv2.putText(
            frame,
            f"MOCK t={elapsed:.1f}s",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        time.sleep(1 / 30)
        return True, frame

    def set(self, *_args, **_kwargs):
        pass

    def release(self):
        self._released = True


def _open_camera():
    if os.environ.get("MOCK_CAMERA") == "1":
        logger.info("MOCK_CAMERA=1 — using synthesized frame source")
        return _FakeVideoCapture()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SHAPE[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SHAPE[0])
    return cap


class FrontendState:
    """Shared between the MJPEG generator and the control endpoints.

    The lock guards transport swaps so the video generator never sees a half-closed client.
    """

    def __init__(self):
        self.cap: cv2.VideoCapture | None = None
        self.client: Transport | None = None
        self.active_transport: str | None = None
        self.infer: bool = False
        self.bench_results: dict = {}
        self.lock = threading.Lock()

    def ensure_camera(self):
        if self.cap is None:
            logger.info("Start camera initialization")
            self.cap = _open_camera()
            logger.info("Camera initialized")

    def set_control(self, transport_name: str, infer: bool) -> None:
        with self.lock:
            self.infer = infer
            if not infer:
                self._disconnect_locked()
                return

            if self.active_transport == transport_name and self.client is not None:
                return

            # Ask the server to switch to the requested transport first.
            self._disconnect_locked()
            resp = requests.post(f"{CONTROL_BASE}/switch", json={"name": transport_name}, timeout=CONTROL_TIMEOUT_S)
            resp.raise_for_status()
            port = resp.json()["port"]

            cls = registry.get(transport_name)
            client = cls()
            client.connect(HOST, port)
            self.client = client
            self.active_transport = transport_name
            logger.info(f"client connected: {transport_name} → :{port}")

    def _disconnect_locked(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                logger.exception("failed to disconnect current client")
            self.client = None
            self.active_transport = None

    def record_timing(self, transport_name: str, timings: dict) -> None:
        server_ms = timings.get("decode_ms", 0.0) + timings.get("infer_ms", 0.0) + timings.get("post_ms", 0.0)
        comms_ms = max(0.0, timings.get("total_ms", 0.0) - timings.get("encode_ms", 0.0) - server_ms)
        transmission_ms = max(0.0, timings.get("total_ms", 0.0) - timings.get("infer_ms", 0.0))
        timings = {**timings, "comms_ms": comms_ms, "transmission_ms": transmission_ms}

        bench = self.bench_results.setdefault(transport_name, {"active_time_s": 0.0, **{col: [] for col in TIMING_COLUMNS}})
        for col in TIMING_COLUMNS:
            if col in timings:
                bench[col].append(timings[col])
        bench["active_time_s"] += timings.get("total_ms", 0.0) / 1000

    def build_stats_rows(self) -> list[dict]:
        rows = []
        for name, data in self.bench_results.items():
            totals = data["total_ms"]
            if not totals:
                continue
            duration_s = data["active_time_s"]
            row = {
                "Backend": name,
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


def _mjpeg_frames():
    """Blocking generator: captures frames, runs inference when enabled, yields MJPEG parts."""
    state.ensure_camera()
    assert state.cap is not None

    while True:
        ret, frame = state.cap.read()
        if not ret:
            logger.warning("failed to capture frame")
            time.sleep(0.01)
            continue

        with state.lock:
            active_client = state.client
            active_transport = state.active_transport
            infer = state.infer

        if infer and active_client is not None and active_transport is not None:
            detections, timings = active_client.send(frame)
            state.record_timing(active_transport, timings)
            fps = 1000 / timings["total_ms"] if timings.get("total_ms", 0) > 0 else 0
            if detections is not None:
                frame = draw_detections(frame, detections)
                frame = draw_fps(frame, fps)

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(_mjpeg_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=UI_PORT)
