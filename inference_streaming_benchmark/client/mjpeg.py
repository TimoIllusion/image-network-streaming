from __future__ import annotations

import time

import cv2

from .processor import FrameProcessor

MJPEG_FRAME_INTERVAL_S = 1 / 30


def _mjpeg_frames(processor: FrameProcessor):
    """MJPEG generator — passive observer of the FrameProcessor's latest annotated frame."""
    processor.viewer_started()
    try:
        while True:
            frame = processor.latest_frame()
            if frame is not None:
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            time.sleep(MJPEG_FRAME_INTERVAL_S)
    finally:
        processor.viewer_stopped()
