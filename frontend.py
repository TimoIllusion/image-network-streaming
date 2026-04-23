import os

import streamlit as st

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time

import cv2

from inference_streaming_benchmark.backend.api import BackendInterface
from inference_streaming_benchmark.frontend.media import draw_detections, draw_fps
from inference_streaming_benchmark.logging import logger

BACKEND_NAMES = ["fastapi", "zmq", "imagezmq", "grpc"]


def _create_backend(name: str) -> BackendInterface:
    if name == "fastapi":
        from inference_streaming_benchmark.backend.fastapi.api import FastAPIBackendInterface

        return FastAPIBackendInterface()
    if name == "zmq":
        from inference_streaming_benchmark.backend.zmq.api import ZMQBackendInterface

        return ZMQBackendInterface()
    if name == "imagezmq":
        from inference_streaming_benchmark.backend.imagezmq.api import ImageZMQBackendInterface

        return ImageZMQBackendInterface()
    if name == "grpc":
        from inference_streaming_benchmark.backend.grpc.api import GRPCBackendInterface

        return GRPCBackendInterface()
    raise ValueError(f"Invalid backend: {name}")


selected = st.selectbox("Backend", BACKEND_NAMES)

st.title(f"Webcam Object Detection using {selected.upper()}")
infer = st.checkbox("Enable object detection")
FRAME_WINDOW = st.image([])
TEXT_MESSAGE = st.empty()

st.session_state.setdefault("cap", None)
st.session_state.setdefault("backend_interface", None)
st.session_state.setdefault("active_backend_name", None)

logger.debug("STREAMLIT APP ITERATION")
logger.debug(st.session_state)
logger.debug("INFER: {}", infer)
logger.debug("SELECTED: {}", selected)

current = st.session_state["backend_interface"]
active = st.session_state["active_backend_name"]

if not infer:
    if current is not None:
        current.close()
        st.session_state["backend_interface"] = None
        st.session_state["active_backend_name"] = None
        logger.info("Backend comms closed")
else:
    if current is None or active != selected:
        if current is not None:
            current.close()
            logger.info("Backend comms closed (switching to {})", selected)
        st.session_state["backend_interface"] = _create_backend(selected)
        st.session_state["active_backend_name"] = selected
        logger.info("Backend comms initialized: {}", selected)

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

    # Send frame to server for object detection
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

    # Display captured frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")
