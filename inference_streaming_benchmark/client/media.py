import cv2
import numpy as np


def draw_detections(img: np.ndarray, detections: list[dict]):

    h, w, _ = img.shape

    scale = w / 640

    for obj_detection in detections:
        cv2.rectangle(
            img,
            (int(obj_detection["box"]["x1"]), int(obj_detection["box"]["y1"])),
            (int(obj_detection["box"]["x2"]), int(obj_detection["box"]["y2"])),
            (0, 255, 0),
            2,
        )

        # draw class name, id and confidence above rectangle
        text_x = int(obj_detection["box"]["x1"])
        text_y = int(obj_detection["box"]["y1"]) - 5

        cv2.putText(
            img,
            f"{obj_detection['name']} {obj_detection['class']} {obj_detection['confidence']:0.2f}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5 * scale,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return img


def draw_fps(img: np.ndarray, fps: float):

    h, w, _ = img.shape

    scale = w / 640

    x = 10
    y = int(40 * scale)

    cv2.putText(
        img,
        f"FPS: {fps:0.2f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return img
