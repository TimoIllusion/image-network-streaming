import zmq
import json
import time
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io

# Initialize logging
from image_network_streaming.logging import logger

# Load the YOLO model
model = YOLO("yolov8n.pt")


def detect(image_data):
    """
    Perform object detection on the image data.
    """
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)

    # Ensure image is in RGB format if it's a PNG with an alpha channel
    if image.shape[-1] == 4:
        image = image[..., :3]

    # Perform inference
    results = model(image)

    # Process results list
    results_prepared_for_transmission = []
    for result in results:
        result_json_str = result.tojson()
        # Convert back to dict object
        result_py = json.loads(result_json_str)
        results_prepared_for_transmission.append(result_py)

    return results_prepared_for_transmission


def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP socket for reply
    socket.bind("tcp://*:5555")

    logger.info("ZMQ context and socket initialized.")

    logger.info("Loop for receiving images and sending back detections started.")

    while True:
        logger.info("Waiting for image data...")
        #  Wait for next request from client
        image_data = socket.recv()

        t0 = time.time()

        detections = detect(image_data)

        t1 = time.time()

        logger.info(f"Detection time: {t1 - t0}")
        logger.info(detections)

        #  Send reply back to client
        logger.info("Sending back detections...")
        socket.send_json({"batched_detections": detections})
