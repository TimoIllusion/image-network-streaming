from __future__ import annotations

import json

import cv2
import numpy as np

from inference_streaming_benchmark.engine import decode_jpeg_bytes

FRAME_SHAPE: tuple[int, int, int] = (1080, 1920, 3)
RAW_MAGIC = b"INFSBRAW1 "


def encode(frame: np.ndarray, raw: bool) -> bytes:
    if raw:
        header = {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
        }
        return RAW_MAGIC + json.dumps(header, separators=(",", ":")).encode("ascii") + b"\n" + frame.tobytes()
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def decode(data: bytes, raw: bool) -> np.ndarray:
    if raw:
        shape = FRAME_SHAPE
        dtype = np.dtype(np.uint8)
        payload = data
        if data.startswith(RAW_MAGIC):
            header_bytes, payload = data[len(RAW_MAGIC) :].split(b"\n", 1)
            header = json.loads(header_bytes.decode("ascii"))
            shape = tuple(int(dim) for dim in header["shape"])
            dtype = np.dtype(header.get("dtype", "uint8"))
        return np.frombuffer(payload, dtype=dtype).reshape(shape)
    return decode_jpeg_bytes(data)
