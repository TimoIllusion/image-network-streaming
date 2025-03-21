import imagezmq
import json
import time
from ultralytics import YOLO

# Initialize logging
from image_network_streaming.logging import logger

# Load the YOLO model
model = YOLO("yolov8n.pt")


def detect(image):
    """
    Perform object detection on the image data.
    """

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
    image_hub = imagezmq.ImageHub(open_port="tcp://0.0.0.0:5555")

    logger.info("ImageZMQ HUB initialized.")

    logger.info("Loop for receiving images and sending back detections started.")

    while True:
        logger.info("Waiting for image data...")
        #  Wait for next request from client
        text, image = image_hub.recv_image()

        t0 = time.time()

        detections = detect(image)

        t1 = time.time()

        logger.info(f"Detection time: {t1 - t0}")
        logger.info(detections)

        #  Send reply back to client
        logger.info("Sending back detections...")

        detections_json_str = json.dumps({"batched_detections": detections})
        image_hub.send_reply(bytes(detections_json_str, "utf-8"))
