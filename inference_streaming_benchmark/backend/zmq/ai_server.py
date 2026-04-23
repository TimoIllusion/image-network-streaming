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

        t0 = time.perf_counter()
        image = decode_jpeg_bytes(image_data)
        decode_ms = (time.perf_counter() - t0) * 1000
        detections, timings = run_inference(image)
        timings["decode_ms"] = decode_ms

        logger.info(f"Detection time: {timings['infer_ms']:.1f}ms")

        socket.send_json({"batched_detections": detections, "timings": timings})
