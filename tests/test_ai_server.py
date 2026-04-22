import pytest
import json
import io

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark.backend.fastapi.ai_server import app  # noqa: E402
from inference_streaming_benchmark.logging import logger  # noqa: E402


def test_end_to_end(monkeypatch):
    """
    Deterministic end-to-end test:
    - No filesystem resources required
    - No webcam required
    - No Ultralytics weights download / GPU required (model is stubbed)
    """

    # Import the module (not just app) so we can monkeypatch its globals.
    import inference_streaming_benchmark.backend.fastapi.ai_server as ai_server

    class _FakeResult:
        def to_json(self) -> str:
            # Ultralytics returns a JSON string; we emulate a minimal structure
            # compatible with frontend drawing code.
            return json.dumps(
                [
                    {
                        "box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40},
                        "confidence": 0.99,
                        "class": 0,
                        "name": "object",
                    }
                ]
            )

    def _fake_model(_image):
        return [_FakeResult()]

    async def _run_sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(ai_server, "model", _fake_model)
    monkeypatch.setattr(ai_server, "run_in_threadpool", _run_sync)

    client = TestClient(app)

    # Create a tiny deterministic image in-memory.
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
