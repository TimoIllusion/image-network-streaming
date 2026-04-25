import numpy as np
import pytest

pytest.importorskip("websockets")
pytest.importorskip("numpy")

from inference_streaming_benchmark import transports as _transports  # noqa: F401 — triggers registration
from inference_streaming_benchmark.transports import registry
from inference_streaming_benchmark.transports.websocket.transport import WebSocketTransport

WebSocketRawTransport = registry.get("websocket_raw")


@pytest.mark.parametrize(
    "cls,port",
    [
        (WebSocketTransport, 19099),
        (WebSocketRawTransport, 19100),
    ],
    ids=["jpeg", "raw"],
)
def test_end_to_end(cls, port, monkeypatch):
    fake_detections = [[{"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.99, "class": 0, "name": "object"}]]

    def fake_handler(_image):
        return fake_detections, {"infer_ms": 1.0, "post_ms": 0.1}

    import inference_streaming_benchmark.transports.codec as codec_mod

    monkeypatch.setattr(codec_mod, "FRAME_SHAPE", (4, 4, 3))

    server = cls()
    server.start(port, fake_handler)  # blocks until ready
    try:
        client = cls()
        client.connect("127.0.0.1", port)
        try:
            frame = np.zeros((4, 4, 3), dtype=np.uint8)
            detections, timings = client.send(frame)
            assert detections == fake_detections[0]
            assert "decode_ms" in timings
            assert "infer_ms" in timings
        finally:
            client.disconnect()
    finally:
        server.stop()


def test_persistent_connection_handles_multiple_frames():
    """WebSocket should stay open across multiple send/recv round-trips — that's its advantage over HTTP."""
    fake_detections = [[{"box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "confidence": 0.5, "class": 0, "name": "thing"}]]

    def fake_handler(_image):
        return fake_detections, {"infer_ms": 0.5, "post_ms": 0.05}

    server = WebSocketTransport()
    server.start(19101, fake_handler)
    try:
        client = WebSocketTransport()
        client.connect("127.0.0.1", 19101)
        try:
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            for _ in range(3):
                detections, timings = client.send(frame)
                assert detections == fake_detections[0]
        finally:
            client.disconnect()
    finally:
        server.stop()
