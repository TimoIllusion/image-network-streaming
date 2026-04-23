import json
import time

import imagezmq

from inference_streaming_benchmark.backend.inference import run_inference
from inference_streaming_benchmark.logging import logger


def main(port: int = 5555):
    image_hub = imagezmq.ImageHub(open_port=f"tcp://0.0.0.0:{port}")

    logger.info("ImageZMQ HUB initialized.")
    logger.info("Loop for receiving images and sending back detections started.")

    while True:
        logger.info("Waiting for image data...")
        text, image = image_hub.recv_image()

        t0 = time.perf_counter()
        # ImageZMQ delivers a numpy array directly — no JPEG decode needed on our side.
        decode_ms = (time.perf_counter() - t0) * 1000
        detections, timings = run_inference(image)
        timings["decode_ms"] = decode_ms

        logger.info(f"Detection time: {timings['infer_ms']:.1f}ms")

        image_hub.send_reply(json.dumps({"batched_detections": detections, "timings": timings}).encode())
