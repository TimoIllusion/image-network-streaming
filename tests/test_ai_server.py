import io

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark.backend.http_multipart.ai_server import app  # noqa: E402
from inference_streaming_benchmark.logging import logger  # noqa: E402


def test_end_to_end(monkeypatch):
    """
    Deterministic end-to-end test:
    - No filesystem resources required
    - No webcam required
    - No Ultralytics weights download / GPU required (inference is stubbed)
    """

    import inference_streaming_benchmark.backend.http_multipart.ai_server as ai_server

    _fake_detections = [[{"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.99, "class": 0, "name": "object"}]]

    def _fake_run_inference(_image):
        return _fake_detections, {"infer_ms": 1.0, "post_ms": 0.1}

    async def _run_sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(ai_server, "run_inference", _fake_run_inference)
    monkeypatch.setattr(ai_server, "run_in_threadpool", _run_sync)

    client = TestClient(app)

    from PIL import Image

    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/detect/",
        files={"file": ("example.png", buf.getvalue(), "image/png")},
    )

    assert response.status_code == 200
    logger.info(f"Response: {response.content}")

    data = response.json()

    detections_single = data["batched_detections"][0]

    assert isinstance(detections_single, list)
    assert len(detections_single) == 1
    assert detections_single[0]["box"]["x1"] == 10
