import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("numpy")
pytest.importorskip("PIL")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark.transports.websocket.transport import (  # noqa: E402
    WebSocketRawTransport,
    WebSocketTransport,
)


def _jpeg_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize("cls", [WebSocketTransport, WebSocketRawTransport], ids=["jpeg", "raw"])
def test_end_to_end(cls, monkeypatch):
    """End-to-end test of the WebSocket transport (both JPEG and raw variants)."""
    fake_detections = [[{"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.99, "class": 0, "name": "object"}]]

    def fake_handler(_image):
        return fake_detections, {"infer_ms": 1.0, "post_ms": 0.1}

    async def _run_sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import inference_streaming_benchmark.transports.websocket.transport as mod

    monkeypatch.setattr(mod, "run_in_threadpool", _run_sync)
    # Shrink the raw shape so the raw-variant test payload stays tiny.
    monkeypatch.setattr(mod, "_RAW_SHAPE", (4, 4, 3))

    app = cls.build_app(fake_handler)
    client = TestClient(app)

    with client.websocket_connect("/detect") as ws:
        payload = np.zeros((4, 4, 3), dtype=np.uint8).tobytes() if cls.RAW else _jpeg_bytes()
        ws.send_bytes(payload)
        data = ws.receive_json()

    detections_single = data["batched_detections"][0]
    assert isinstance(detections_single, list)
    assert len(detections_single) == 1
    assert detections_single[0]["box"]["x1"] == 10
    assert "decode_ms" in data["timings"]


def test_persistent_connection_handles_multiple_frames(monkeypatch):
    """The WebSocket should stay open across multiple send/recv round-trips — that's its raison d'être vs HTTP."""
    fake_detections = [[{"box": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}, "confidence": 0.5, "class": 0, "name": "thing"}]]

    def fake_handler(_image):
        return fake_detections, {"infer_ms": 0.5, "post_ms": 0.05}

    async def _run_sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import inference_streaming_benchmark.transports.websocket.transport as mod

    monkeypatch.setattr(mod, "run_in_threadpool", _run_sync)

    app = WebSocketTransport.build_app(fake_handler)
    client = TestClient(app)

    with client.websocket_connect("/detect") as ws:
        jpeg = _jpeg_bytes()
        for _ in range(3):
            ws.send_bytes(jpeg)
            data = ws.receive_json()
            assert data["batched_detections"][0][0]["name"] == "thing"
