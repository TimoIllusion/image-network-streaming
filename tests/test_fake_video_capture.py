import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

import frontend  # noqa: E402
from frontend import _FakeVideoCapture  # noqa: E402


def test_read_returns_bgr_1080p_frame(monkeypatch):
    """MOCK_CAMERA must return 1920x1080 BGR frames so YOLO and the raw-transport shape assumption hold."""
    monkeypatch.setattr(frontend.time, "sleep", lambda _s: None)

    cap = _FakeVideoCapture()
    ok, frame = cap.read()

    assert ok is True
    assert frame.dtype == np.uint8
    assert frame.shape == (1080, 1920, 3)
    # Not all-zero → the image file loaded successfully (catches silent resource loss).
    assert int(frame.sum()) > 0


def test_release_stops_subsequent_reads(monkeypatch):
    monkeypatch.setattr(frontend.time, "sleep", lambda _s: None)

    cap = _FakeVideoCapture()
    cap.release()
    ok, frame = cap.read()

    assert ok is False
    assert frame is None


def test_set_is_noop(monkeypatch):
    """cv2.VideoCapture.set is called by frontend.py to pick resolution — the fake must accept it without error."""
    monkeypatch.setattr(frontend.time, "sleep", lambda _s: None)
    cap = _FakeVideoCapture()
    # Matches the real cv2 API call at frontend.py:88-89
    cap.set(0, 1920)
    cap.set(0, 1080)
    # Still readable afterwards
    ok, _ = cap.read()
    assert ok is True
