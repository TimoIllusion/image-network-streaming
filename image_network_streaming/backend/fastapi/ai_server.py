from typing import List
import json
import time

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


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
    image = Image.open(io.BytesIO(contents))
    image = np.array(image)

    # Ensure image is in RGB format if it's a PNG with an alpha channel
    if image.shape[-1] == 4:
        image = image[..., :3]

    t1 = time.time()

    # Perform inference
    results: List[Results] = model(image)

    logger.info(results)

    t2 = time.time()

    # Process results list
    results_prepared_for_transmission = []
    for result in results:
        result_json_str = result.tojson()
        logger.info(result_json_str)

        # convert back to dict object
        result_py = json.loads(result_json_str)
        logger.info(result_py)

        results_prepared_for_transmission.append(result_py)

    logger.info(results_prepared_for_transmission)

    t3 = time.time()

    logger.info(f"Total time: {t3 - t0}")
    logger.info(f"Read time: {t1 - t0}")
    logger.info(f"Inference time: {t2 - t1}")
    logger.info(f"Results processing time: {t3 - t2}")

    return JSONResponse(
        content={"batched_detections": results_prepared_for_transmission}
    )
