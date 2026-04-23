from __future__ import annotations

import time

from fastapi import FastAPI, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from inference_streaming_benchmark.backend.inference import decode_jpeg_bytes, run_inference
from inference_streaming_benchmark.logging import logger

app = FastAPI()


def _infer_with_decode_timing(contents: bytes) -> tuple[list[dict], dict]:
    t0 = time.perf_counter()
    image = decode_jpeg_bytes(contents)
    decode_ms = (time.perf_counter() - t0) * 1000
    results, timings = run_inference(image)
    timings["decode_ms"] = decode_ms
    return results, timings


@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    t0 = time.perf_counter()

    contents = await file.read()

    # YOLO inference is CPU/GPU bound and synchronous; run it off the event loop so
    # concurrent requests don't stall behind one long inference.
    results, timings = await run_in_threadpool(_infer_with_decode_timing, contents)

    logger.info(
        "detect total=%.1fms decode=%.1fms infer=%.1fms post=%.1fms",
        (time.perf_counter() - t0) * 1000,
        timings["decode_ms"],
        timings["infer_ms"],
        timings["post_ms"],
    )

    return JSONResponse(content={"batched_detections": results, "timings": timings})
