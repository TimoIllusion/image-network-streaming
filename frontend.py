import streamlit as st

import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import time
import cv2
import sys


from image_network_streaming.logging import logger
from image_network_streaming.frontend.media import draw_detections, draw_fps


# construct the argument parser and parse the arguments
backend = sys.argv[1]

# Streamlit webcam input
st.title(f"Webcam Object Detection using {backend.upper()}")
infer = st.checkbox("Enable object detection")
FRAME_WINDOW = st.image([])
TEXT_MESSAGE = st.empty()

# initialize state
if "cap" not in st.session_state:
    st.session_state["cap"] = None

if "backend_interface" not in st.session_state:
    st.session_state["backend_interface"] = None

# for debugging
logger.debug("STREAMLIT APP ITERATION")
logger.debug(st.session_state)
logger.debug("INFER: {}", infer)

# initialize/deactivate comms
logger.debug("BACKEND_INTERFACE: {}", st.session_state["backend_interface"])
if st.session_state["backend_interface"] is not None:
    if not infer:
        st.session_state["backend_interface"].close()
        st.session_state["backend_interface"] = None
        logger.info("Backend comms closed")
    else:
        pass
else:
    if backend == "fastapi":
        from image_network_streaming.backend.fastapi.api import FastAPIBackendInterface

        backend_interface = FastAPIBackendInterface()
    elif backend == "zmq":
        from image_network_streaming.backend.zmq.api import ZMQBackendInterface

        backend_interface = ZMQBackendInterface()
    elif backend == "imagezmq":
        from image_network_streaming.backend.imagezmq.api import (
            ImageZMQBackendInterface,
        )

        backend_interface = ImageZMQBackendInterface()
    elif backend == "grpc":
        from image_network_streaming.backend.grpc.api import GRPCBackendInterface

        backend_interface = GRPCBackendInterface()
    else:
        raise ValueError(f"Invalid backend: {backend}")
    st.session_state["backend_interface"] = backend_interface
    logger.info("Backend comms initialized")

logger.debug(st.session_state)

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
        detection_results_single = st.session_state[
            "backend_interface"
        ].send_frame_to_ai_server(frame)
        inference_duration = time.time() - t0
        fps = 1 / inference_duration

        if detection_results_single is not None:
            # You can now process and display the detection results as needed
            TEXT_MESSAGE.text(detection_results_single)

            frame = draw_detections(frame, detection_results_single)
            frame = draw_fps(frame, fps)
        else:
            TEXT_MESSAGE.text("Failed to get detection results")

    # Display captured frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")
