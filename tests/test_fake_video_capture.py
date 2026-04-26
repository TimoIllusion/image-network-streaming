import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from inference_streaming_benchmark.client import camera as camera_mod  # noqa: E402
from inference_streaming_benchmark.client.camera import _FakeVideoCapture  # noqa: E402


def test_read_returns_bgr_1080p_frame(monkeypatch):
    """MOCK_CAMERA must return 1920x1080 BGR frames so YOLO and the raw-transport shape assumption hold."""
    monkeypatch.setattr(camera_mod.time, "sleep", lambda _s: None)

    cap = _FakeVideoCapture()
    ok, frame = cap.read()

    assert ok is True
    assert frame.dtype == np.uint8
    assert frame.shape == (1080, 1920, 3)
    # Not all-zero → the image file loaded successfully (catches silent resource loss).
    assert int(frame.sum()) > 0


def test_release_stops_subsequent_reads(monkeypatch):
    monkeypatch.setattr(camera_mod.time, "sleep", lambda _s: None)

    cap = _FakeVideoCapture()
    cap.release()
    ok, frame = cap.read()

    assert ok is False
    assert frame is None


def test_set_is_noop(monkeypatch):
    """cv2.VideoCapture.set is called by the camera module to pick resolution — the fake must accept it without error."""
    monkeypatch.setattr(camera_mod.time, "sleep", lambda _s: None)
    cap = _FakeVideoCapture()
    # Matches the real cv2 API call at camera.py
    cap.set(0, 1920)
    cap.set(0, 1080)
    # Still readable afterwards
    ok, _ = cap.read()
    assert ok is True


def test_camera_handle_set_mode_swaps_factory(monkeypatch):
    """CameraHandle.set_mode releases the current cap so ensure() re-opens with the new factory."""
    monkeypatch.setattr(camera_mod.time, "sleep", lambda _s: None)
    pytest.importorskip("fastapi")  # CameraHandle lives in client.state, which imports config

    from inference_streaming_benchmark.client.state import CameraHandle

    sentinel_real = object()

    def _fake_real():
        return _StubCap(sentinel_real)

    monkeypatch.setattr(camera_mod, "_open_real_camera", _fake_real)

    handle = CameraHandle(initial_mode="real")
    cap1 = handle.ensure()
    assert cap1.tag is sentinel_real

    handle.set_mode("mock")
    assert handle.mode == "mock"
    cap2 = handle.ensure()
    # Mode switch must yield a different cap (real → mock).
    assert cap2 is not cap1
    assert isinstance(cap2, _FakeVideoCapture)


def test_camera_handle_set_mode_rejects_unknown():
    pytest.importorskip("fastapi")
    from inference_streaming_benchmark.client.state import CameraHandle

    handle = CameraHandle(initial_mode="real")
    try:
        handle.set_mode("bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown mode")


class _StubCap:
    def __init__(self, tag):
        self.tag = tag
        self._released = False

    def read(self):
        return True, None

    def release(self):
        self._released = True

    def set(self, *_a, **_k):
        pass
