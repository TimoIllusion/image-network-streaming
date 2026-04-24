import pytest

pytest.importorskip("fastapi")
pytest.importorskip("numpy")

import numpy as np  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark.transports.http_multipart.transport import (  # noqa: E402
    HTTPMultipartRawTransport,
)


def test_end_to_end_raw(monkeypatch):
    """End-to-end test of the raw-ndarray HTTP variant: POST application/octet-stream body."""
    fake_detections = [[{"box": {"x1": 10, "y1": 20, "x2": 30, "y2": 40}, "confidence": 0.99, "class": 0, "name": "object"}]]

    def fake_handler(_image):
        return fake_detections, {"infer_ms": 1.0, "post_ms": 0.1}

    async def _run_sync(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    import inference_streaming_benchmark.transports.http_multipart.transport as mod

    monkeypatch.setattr(mod, "run_in_threadpool", _run_sync)
    # Shrink the raw shape so the test payload is tiny instead of 6 MB of zeros.
    monkeypatch.setattr(mod, "_RAW_SHAPE", (4, 4, 3))

    app = HTTPMultipartRawTransport.build_app(fake_handler)
    client = TestClient(app)

    payload = np.zeros((4, 4, 3), dtype=np.uint8).tobytes()
    response = client.post(
        "/detect/",
        content=payload,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200
    data = response.json()
    detections_single = data["batched_detections"][0]
    assert isinstance(detections_single, list)
    assert len(detections_single) == 1
    assert detections_single[0]["box"]["x1"] == 10
    # decode_ms (reshape) should be present and tiny.
    assert "decode_ms" in data["timings"]
