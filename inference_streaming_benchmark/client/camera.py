from __future__ import annotations

import os
import time
from pathlib import Path

import cv2

from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports.codec import FRAME_SHAPE

_MOCK_IMAGE_PATH = Path(__file__).parents[2] / "resources" / "example_dall_e.png"

CAMERA_MODES = ("real", "mock")


class _FakeVideoCapture:
    """Static 1920×1080 BGR frames for headless clients and Claude-driven smoke tests.

    Returns the resources/example_dall_e.png image (resized to 1920×1080) with a
    timestamp overlay at ~30fps. The image contains people, chairs, a dining table,
    and a laptop so YOLO produces real detections.
    """

    def __init__(self):
        self._t0 = time.time()
        self._released = False
        img = cv2.imread(str(_MOCK_IMAGE_PATH))
        if img is None:
            raise RuntimeError(f"mock camera: could not load {_MOCK_IMAGE_PATH}")
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


def _open_real_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SHAPE[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SHAPE[0])
    return cap


def _open_mock_camera():
    return _FakeVideoCapture()


def open_camera(mode: str):
    if mode == "mock":
        logger.info("camera: mock (synthetic frames)")
        return _open_mock_camera()
    if mode == "real":
        logger.info("camera: real (cv2.VideoCapture)")
        return _open_real_camera()
    raise ValueError(f"unknown camera mode: {mode!r} (expected one of {CAMERA_MODES})")


def initial_mode_from_env() -> str:
    return "mock" if os.environ.get("MOCK_CAMERA") == "1" else "real"
