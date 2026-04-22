from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image

from inference_streaming_benchmark.logging import logger

if TYPE_CHECKING:
    from ultralytics.engine.results import Results

app = FastAPI()

# Model is loaded lazily to keep imports lightweight (e.g. in CI/tests) and to
# avoid downloading weights at import time.
model: Callable[[np.ndarray], list[Results]] | None = None


def _get_or_load_model():
    global model
    if model is not None:
        return model

    from ultralytics import YOLO  # imported lazily on purpose

    model = YOLO("yolov8n.pt")
    return model


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):

    t0 = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = np.array(image)

    t1 = time.time()

    # Perform inference
    # YOLO inference is CPU/GPU bound and synchronous; run it off the event loop so
    # concurrent requests don't stall behind one long inference.
    _model = _get_or_load_model()
    results: list[Results] = await run_in_threadpool(_model, image)

    t2 = time.time()

    # Process results list
    results_prepared_for_transmission = []
    for result in results:
        # Ultralytics Results API uses to_json() (not tojson()).
        result_json_str = result.to_json()

        # convert back to dict object
        result_py = json.loads(result_json_str)

        results_prepared_for_transmission.append(result_py)

    t3 = time.time()

    logger.info(
        "detect timings total=%.3fs read=%.3fs infer=%.3fs post=%.3fs",
        (t3 - t0),
        (t1 - t0),
        (t2 - t1),
        (t3 - t2),
    )

    return JSONResponse(content={"batched_detections": results_prepared_for_transmission})
