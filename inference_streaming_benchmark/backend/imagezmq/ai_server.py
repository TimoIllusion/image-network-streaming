import json
import time

import imagezmq

from inference_streaming_benchmark.backend.inference import run_inference
from inference_streaming_benchmark.logging import logger


def main():
    image_hub = imagezmq.ImageHub(open_port="tcp://0.0.0.0:5555")

    logger.info("ImageZMQ HUB initialized.")
    logger.info("Loop for receiving images and sending back detections started.")

    while True:
        logger.info("Waiting for image data...")
        text, image = image_hub.recv_image()

        t0 = time.time()
        detections = run_inference(image)
        t1 = time.time()

        logger.info(f"Detection time: {t1 - t0}")

        image_hub.send_reply(json.dumps({"batched_detections": detections}).encode())
