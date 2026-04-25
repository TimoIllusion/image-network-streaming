from __future__ import annotations

import time

import cv2

from inference_streaming_benchmark.logging import logger

from .media import draw_detections, draw_fps
from .state import FrontendState


def _mjpeg_frames(state: FrontendState):
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
