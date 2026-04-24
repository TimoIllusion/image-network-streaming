from __future__ import annotations

import io
import json
import time

import numpy as np
from PIL import Image

from inference_streaming_benchmark.logging import logger


def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


class InferenceEngine:
    """YOLO-backed detector. Loads the model once on first inference."""

    def __init__(self):
        self._model = None

    def _get_or_load_model(self):
        if self._model is not None:
            return self._model

        import torch
        from ultralytics import YOLO  # lazy — keeps CI imports lightweight

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Loading YOLO model on device: {device}")
        self._model = YOLO("yolov8n.pt")
        self._model.to(device)
        return self._model

    def infer(self, image: np.ndarray) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        results = self._get_or_load_model()(image)
        t1 = time.perf_counter()
        out = [json.loads(result.to_json()) for result in results]
        t2 = time.perf_counter()
        timings = {"infer_ms": (t1 - t0) * 1000, "post_ms": (t2 - t1) * 1000}
        return out, timings
