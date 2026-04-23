import os

import streamlit as st

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time
from urllib.parse import urlparse

import cv2
import requests

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.frontend.media import draw_detections, draw_fps
from inference_streaming_benchmark.logging import logger

DEFAULT_SIDECAR_URL = "http://localhost:9000"


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


def _discover(sidecar_url: str) -> tuple[str, str, int]:
    """Query the sidecar /info endpoint. Returns (transport, host, port)."""
    response = requests.get(f"{sidecar_url.rstrip('/')}/info", timeout=2)
    response.raise_for_status()
    info = response.json()
    host = urlparse(sidecar_url).hostname or "localhost"
    return info["transport"], host, info["port"]


sidecar_url = st.text_input("Backend sidecar URL", DEFAULT_SIDECAR_URL)

st.session_state.setdefault("cap", None)
st.session_state.setdefault("backend_interface", None)
st.session_state.setdefault("active_sidecar_url", None)
st.session_state.setdefault("active_transport", None)

# Try to discover whenever the URL changes.
if sidecar_url != st.session_state["active_sidecar_url"]:
    # URL changed → close any existing client, drop cached discovery.
    if st.session_state["backend_interface"] is not None:
        st.session_state["backend_interface"].close()
        st.session_state["backend_interface"] = None
    st.session_state["active_sidecar_url"] = sidecar_url
    st.session_state["active_transport"] = None

if st.session_state["active_transport"] is None:
    try:
        transport, host, port = _discover(sidecar_url)
        st.session_state["active_transport"] = transport
        st.session_state["active_host"] = host
        st.session_state["active_port"] = port
        logger.info(f"Discovered {transport} at {host}:{port}")
    except Exception as e:
        st.error(f"Could not reach sidecar at {sidecar_url}: {e}")
        st.stop()

transport = st.session_state["active_transport"]
host = st.session_state["active_host"]
port = st.session_state["active_port"]

st.title(f"Webcam Object Detection using {transport.upper()}")
st.caption(f"Data plane: {host}:{port}")

infer = st.checkbox("Enable object detection")
FRAME_WINDOW = st.image([])
TEXT_MESSAGE = st.empty()

logger.debug("STREAMLIT APP ITERATION")
logger.debug(st.session_state)
logger.debug("INFER: {}", infer)

current = st.session_state["backend_interface"]

if not infer:
    if current is not None:
        current.close()
        st.session_state["backend_interface"] = None
        logger.info("Backend comms closed")
else:
    if current is None:
        st.session_state["backend_interface"] = _create_backend(transport, host, port)
        logger.info(f"Backend comms initialized: {transport}")

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
        inference_duration = time.time() - t0
        fps = 1 / inference_duration

        if detection_results_single is not None:
            TEXT_MESSAGE.text(detection_results_single)

            frame = draw_detections(frame, detection_results_single)
            frame = draw_fps(frame, fps)
        else:
            TEXT_MESSAGE.text("Failed to get detection results")

    FRAME_WINDOW.image(frame, channels="BGR")
