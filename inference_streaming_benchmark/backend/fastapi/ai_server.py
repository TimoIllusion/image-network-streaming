from __future__ import annotations

import time

from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from inference_streaming_benchmark.backend.inference import decode_jpeg_bytes, run_inference
from inference_streaming_benchmark.logging import logger

app = FastAPI()


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()

    contents = await file.read()
    image = decode_jpeg_bytes(contents)

    t1 = time.time()

    # YOLO inference is CPU/GPU bound and synchronous; run it off the event loop so
    # concurrent requests don't stall behind one long inference.
    results = await run_in_threadpool(run_inference, image)

    t2 = time.time()

    logger.info(
        "detect timings total=%.3fs decode=%.3fs infer=%.3fs",
        (t2 - t0),
        (t1 - t0),
        (t2 - t1),
    )

    return JSONResponse(content={"batched_detections": results})
