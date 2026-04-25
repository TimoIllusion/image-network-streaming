from __future__ import annotations

import cv2
import numpy as np

from inference_streaming_benchmark.engine import decode_jpeg_bytes

FRAME_SHAPE: tuple[int, int, int] = (1080, 1920, 3)


def encode(frame: np.ndarray, raw: bool) -> bytes:
    if raw:
        return frame.tobytes()
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def decode(data: bytes, raw: bool) -> np.ndarray:
    if raw:
        return np.frombuffer(data, dtype=np.uint8).reshape(FRAME_SHAPE)
    return decode_jpeg_bytes(data)
