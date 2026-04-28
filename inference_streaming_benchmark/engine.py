from __future__ import annotations

import io
import json
import queue
import threading
import time

import numpy as np
from PIL import Image

from inference_streaming_benchmark.logging import logger

INFER_MODES = ("single", "unsafe-multi", "multi-instance")


def decode_jpeg_bytes(data: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


class InferenceEngine:
    """YOLO-backed detector. Loads the model once on first inference."""

    def __init__(self, *, mode: str = "single", instances: int = 2):
        self._mode = self._validate_mode(mode)
        self._instances = self._validate_instances(instances)
        self._model = None
        self._instance_pool: dict | None = None
        self._cfg_lock = threading.Lock()
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._reset_instance_pool()

    @staticmethod
    def _validate_mode(mode: str) -> str:
        if mode not in INFER_MODES:
            raise ValueError(f"inference mode must be one of {', '.join(INFER_MODES)}")
        return mode

    @staticmethod
    def _validate_instances(instances: int) -> int:
        if int(instances) < 1:
            raise ValueError("instances must be >= 1")
        return int(instances)

    def state(self) -> dict:
        with self._cfg_lock:
            return {"mode": self._mode, "instances": self._instances}

    def configure(self, *, mode: str | None = None, instances: int | None = None) -> dict:
        with self._cfg_lock:
            if mode is not None:
                self._mode = self._validate_mode(mode)
            if instances is not None:
                new_instances = self._validate_instances(instances)
                if new_instances != self._instances:
                    self._instances = new_instances
                    self._reset_instance_pool()
            return {"mode": self._mode, "instances": self._instances}

    def _reset_instance_pool(self) -> None:
        available_instances: queue.Queue[int] = queue.Queue()
        for idx in range(self._instances):
            available_instances.put(idx)
        self._instance_pool = {
            "models": [None] * self._instances,
            "load_locks": [threading.Lock() for _ in range(self._instances)],
            "available": available_instances,
        }

    def _load_model(self):
        import torch
        from ultralytics import YOLO  # lazy — keeps CI imports lightweight

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        logger.info(f"Loading YOLO model on device: {device}")
        model = YOLO("yolov8n.pt")
        model.to(device)
        return model

    def _get_or_load_model(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            model = self._load_model()
            self._model = model
            return self._model

    def _get_or_load_instance_model(self, idx: int, pool: dict):
        models = pool["models"]
        load_locks = pool["load_locks"]
        if models[idx] is not None:
            return models[idx]
        with load_locks[idx]:
            if models[idx] is not None:
                return models[idx]
            model = self._load_model()
            models[idx] = model
            return model

    def _call_model(self, payload):
        with self._cfg_lock:
            mode = self._mode
            instances = self._instances
            instance_pool = self._instance_pool

        if mode == "single":
            with self._lock:
                return self._get_or_load_model()(payload, verbose=False), mode, instances
        if mode == "unsafe-multi":
            return self._get_or_load_model()(payload, verbose=False), mode, instances

        assert instance_pool is not None
        available_instances = instance_pool["available"]
        idx = available_instances.get()
        try:
            model = self._get_or_load_instance_model(idx, instance_pool)
            return model(payload, verbose=False), mode, instances
        finally:
            available_instances.put(idx)

    @staticmethod
    def _mode_timings(mode: str, instances: int) -> dict:
        return {
            "inference_mode": mode,
            "inference_instances": instances,
        }

    def infer(self, image: np.ndarray) -> tuple[list[dict], dict]:
        t0 = time.perf_counter()
        results, mode, instances = self._call_model(image)
        t1 = time.perf_counter()
        out = [json.loads(result.to_json()) for result in results]
        t2 = time.perf_counter()
        timings = {
            "infer_ms": (t1 - t0) * 1000,
            "post_ms": (t2 - t1) * 1000,
            **self._mode_timings(mode, instances),
        }
        return out, timings

    def infer_batch(self, images: list[np.ndarray]) -> list[tuple[list, dict]]:
        """Run one model call on a batch and return per-frame (detections, timings).

        Per-frame `detections` keeps the same shape as `infer()` returns for a single
        image (a length-1 list wrapping the per-image detection list) so the wire
        envelope's `batched_detections` field is unchanged.

        `infer_ms` is the same wall-clock value for every frame in the batch — that's
        the latency each caller actually waited for the model. `post_ms` is per-frame
        (its own JSON serialization cost).
        """
        if not images:
            return []
        t0 = time.perf_counter()
        results, mode, instances = self._call_model(images)
        t1 = time.perf_counter()
        infer_ms = (t1 - t0) * 1000

        out: list[tuple[list, dict]] = []
        for result in results:
            tp0 = time.perf_counter()
            detections = [json.loads(result.to_json())]
            tp1 = time.perf_counter()
            out.append(
                (
                    detections,
                    {
                        "infer_ms": infer_ms,
                        "post_ms": (tp1 - tp0) * 1000,
                        **self._mode_timings(mode, instances),
                    },
                )
            )
        return out
