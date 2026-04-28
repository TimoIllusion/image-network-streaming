import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark.client.app import build_client_app  # noqa: E402
from inference_streaming_benchmark.client.state import BenchmarkCollector, CameraHandle, TransportSession  # noqa: E402


class _Processor:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class _Registrar:
    def __init__(self, **_kwargs):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_build_client_app_uses_injected_state():
    camera = CameraHandle(initial_mode="mock")
    session = TransportSession()
    collector = BenchmarkCollector()
    processor = _Processor()
    registrar = _Registrar()
    app = build_client_app(
        camera=camera,
        session=session,
        collector=collector,
        processor=processor,
        registrar_factory=lambda **_kwargs: registrar,
    )

    assert app.state.camera is camera
    assert app.state.session is session
    assert app.state.collector is collector
    assert app.state.processor is processor

    with TestClient(app) as client:
        assert processor.started is True
        assert registrar.started is True
        response = client.get("/api/state")
        assert response.status_code == 200
        assert response.json()["mock_camera"] is True

    assert processor.stopped is True
    assert registrar.stopped is True
