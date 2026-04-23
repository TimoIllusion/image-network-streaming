import os

import streamlit as st

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import statistics
import time

import cv2
import requests

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.frontend.media import draw_detections, draw_fps
from inference_streaming_benchmark.logging import logger

# Hardcoded transport registry: name → (data port, sidecar port).
# Host is assumed to be localhost for the status probe and the client.
TRANSPORTS: dict[str, dict[str, int]] = {
    "fastapi": {"data_port": 8008, "sidecar_port": 9001},
    "zmq": {"data_port": 5555, "sidecar_port": 9002},
    "imagezmq": {"data_port": 5556, "sidecar_port": 9003},
    "grpc": {"data_port": 50051, "sidecar_port": 9004},
}

HOST = "localhost"
HEALTH_TIMEOUT_S = 0.2
STATUS_CACHE_TTL_S = 3.0
STATS_REFRESH_EVERY_N_FRAMES = 5


def _create_backend(transport: str, host: str, port: int) -> BackendInterface:
    if transport == "fastapi":
        from inference_streaming_benchmark.backend.fastapi.api import FastAPIBackendInterface

        return FastAPIBackendInterface(host=host, port=port)
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


def _get_statuses() -> dict[str, bool]:
    cached_at = st.session_state.get("status_cached_at", 0.0)
    if time.time() - cached_at < STATUS_CACHE_TTL_S and "statuses" in st.session_state:
        return st.session_state["statuses"]
    statuses = {name: _probe_sidecar(cfg["sidecar_port"]) for name, cfg in TRANSPORTS.items()}
    st.session_state["statuses"] = statuses
    st.session_state["status_cached_at"] = time.time()
    return statuses


def _build_stats_rows() -> list[dict]:
    rows = []
    for backend, data in st.session_state["bench_results"].items():
        lats = data["latencies_ms"]
        if not lats:
            continue
        duration_s = data["active_time_s"]
        sorted_lats = sorted(lats)
        p95_idx = min(int(len(lats) * 0.95), len(lats) - 1)
        rows.append(
            {
                "Backend": backend,
                "Frames": len(lats),
                "Duration (s)": f"{duration_s:.1f}",
                "FPS (mean)": f"{len(lats) / duration_s:.1f}" if duration_s > 0 else "-",
                "Latency p50 (ms)": f"{statistics.median(lats):.1f}",
                "Latency p95 (ms)": f"{sorted_lats[p95_idx]:.1f}",
            }
        )
    return rows


def _render_stats(placeholder) -> None:
    rows = _build_stats_rows()
    if rows:
        placeholder.table(rows)
    else:
        placeholder.caption("No measurements yet — enable object detection to start collecting.")


statuses = _get_statuses()
option_labels = {name: f"{name} {'✅' if ok else '❌ offline'}" for name, ok in statuses.items()}

selected = st.selectbox("Backend", list(TRANSPORTS.keys()), format_func=lambda n: option_labels[n])

st.session_state.setdefault("bench_results", {})

stats_placeholder = st.empty()
if st.button("Clear results"):
    st.session_state["bench_results"] = {}
_render_stats(stats_placeholder)

st.title(f"Webcam Object Detection using {selected.upper()}")
infer = st.checkbox("Enable object detection")
FRAME_WINDOW = st.image([])
TEXT_MESSAGE = st.empty()

st.session_state.setdefault("cap", None)
st.session_state.setdefault("backend_interface", None)
st.session_state.setdefault("active_transport", None)

logger.debug("STREAMLIT APP ITERATION")
logger.debug(st.session_state)
logger.debug("INFER: {}", infer)
logger.debug("SELECTED: {}", selected)

current = st.session_state["backend_interface"]
active = st.session_state["active_transport"]

if not infer:
    if current is not None:
        current.close()
        st.session_state["backend_interface"] = None
        st.session_state["active_transport"] = None
        logger.info("Backend comms closed")
else:
    if not statuses.get(selected, False):
        st.error(f"Selected backend '{selected}' is offline. Start it and try again.")
        st.stop()
    if current is None or active != selected:
        if current is not None:
            current.close()
            logger.info("Backend comms closed (switching to {})", selected)
        data_port = TRANSPORTS[selected]["data_port"]
        st.session_state["backend_interface"] = _create_backend(selected, HOST, data_port)
        st.session_state["active_transport"] = selected
        logger.info(f"Backend comms initialized: {selected}")

# initialize camera
if st.session_state["cap"] is None:
    TEXT_MESSAGE.text("Initializing camera...")
    logger.info("Start camera initialization")
    cap = cv2.VideoCapture(0)  # Use 0 for the primary webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    st.session_state["cap"] = cap
    logger.info("Camera initialized")
else:
    cap = st.session_state["cap"]

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame")
        continue

    if infer:
        t0 = time.time()
        detection_results_single = st.session_state["backend_interface"].send_frame_to_ai_server(frame)
        latency_s = time.time() - t0
        latency_ms = latency_s * 1000
        fps = 1 / latency_s if latency_s > 0 else 0

        bench = st.session_state["bench_results"].setdefault(selected, {"latencies_ms": [], "active_time_s": 0.0})
        bench["latencies_ms"].append(latency_ms)
        bench["active_time_s"] += latency_s

        if len(bench["latencies_ms"]) % STATS_REFRESH_EVERY_N_FRAMES == 0:
            _render_stats(stats_placeholder)

        if detection_results_single is not None:
            TEXT_MESSAGE.text(detection_results_single)

            frame = draw_detections(frame, detection_results_single)
            frame = draw_fps(frame, fps)
        else:
            TEXT_MESSAGE.text("Failed to get detection results")

    FRAME_WINDOW.image(frame, channels="BGR")
