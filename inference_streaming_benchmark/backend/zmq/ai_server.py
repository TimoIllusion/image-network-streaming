import time

import zmq

from inference_streaming_benchmark.backend.inference import decode_jpeg_bytes, run_inference
from inference_streaming_benchmark.logging import logger


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    logger.info("ZMQ context and socket initialized.")
    logger.info("Loop for receiving images and sending back detections started.")

    while True:
        logger.info("Waiting for image data...")
        image_data = socket.recv()

        t0 = time.time()
        image = decode_jpeg_bytes(image_data)
        detections = run_inference(image)
        t1 = time.time()

        logger.info(f"Detection time: {t1 - t0}")

        socket.send_json({"batched_detections": detections})
