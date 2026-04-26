import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from inference_streaming_benchmark.client.media import draw_detections, draw_fps


def test_draw_helpers_are_deterministic():
    # Synthetic frame (no webcam dependency)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detections = [
        {
            "box": {"x1": 10, "y1": 20, "x2": 100, "y2": 200},
            "confidence": 0.9,
            "class": 1,
            "name": "obj",
        }
    ]

    out1 = draw_detections(frame.copy(), detections)
    out2 = draw_fps(out1.copy(), fps=12.34)

    assert out2.shape == frame.shape
    # Pixels should have changed (rectangle/text drawn)
    assert int(out2.sum()) > 0
