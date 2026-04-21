from typing import List
import json
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool


from ultralytics import YOLO
from ultralytics.engine.results import Results
import io
from PIL import Image
import numpy as np

from image_network_streaming.logging import logger

app = FastAPI()

# Load the model (specify the path to your YOLOv8 model or use the default one)
model = YOLO("yolov8n.pt")


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
    results: List[Results] = await run_in_threadpool(model, image)

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

    return JSONResponse(
        content={"batched_detections": results_prepared_for_transmission}
    )
