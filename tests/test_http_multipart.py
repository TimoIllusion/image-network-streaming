import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("PIL")

from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark.transports.http_multipart.transport import (  # noqa: E402
    HTTPMultipartTransport,
)


def test_end_to_end(monkeypatch):
    """
    Deterministic end-to-end test through the FastAPI app the transport builds:
    no webcam, no Ultralytics, no network — handler is injected.
    """
    fake_detections = [[{"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.99, "class": 0, "name": "object"}]]

    seen = {}

    def fake_handler(request):
        seen["client_name"] = request.client_name
        seen["request_id"] = request.request_id
        seen["transport"] = request.transport
        return fake_detections, {"infer_ms": 1.0, "post_ms": 0.1}

    async def _run_sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    # Patch the module so the FastAPI threadpool runs inline (keeps the test sync).
    import inference_streaming_benchmark.transports.http_multipart.transport as mod

    monkeypatch.setattr(mod, "run_in_threadpool", _run_sync)

    app = HTTPMultipartTransport.build_app(fake_handler)
    client = TestClient(app)

    from PIL import Image

    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    response = client.post(
        "/detect/",
        files={"file": ("example.png", buf.getvalue(), "image/png")},
        headers={"X-INFSB-Client": "client-http", "X-INFSB-Request-ID": "req-http"},
    )

    assert response.status_code == 200
    data = response.json()
    detections_single = data["batched_detections"][0]
    assert isinstance(detections_single, list)
    assert len(detections_single) == 1
    assert detections_single[0]["box"]["x1"] == 10
    assert seen == {"client_name": "client-http", "request_id": "req-http", "transport": "http_multipart"}
