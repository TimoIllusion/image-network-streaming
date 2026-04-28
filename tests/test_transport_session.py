import threading
import time

import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")

from inference_streaming_benchmark.client.state import TransportSession  # noqa: E402


class _FailingTransport:
    def connect(self, _host, _port):
        raise RuntimeError("connect failed")


class _BlockingClient:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.disconnected = threading.Event()

    def send(self, _frame, *, client_name, request_id):
        self.started.set()
        self.release.wait(timeout=2.0)
        return [], {"total_ms": 1.0}

    def disconnect(self):
        self.disconnected.set()


def test_disconnect_waits_for_in_flight_send():
    session = TransportSession()
    client = _BlockingClient()
    session.client = client
    session.active_transport = "imagezmq"
    session.infer = True

    send_result = {}
    send_thread = threading.Thread(
        target=lambda: send_result.setdefault(
            "value",
            session.send("frame", client_name="client", request_id="req"),
        )
    )
    send_thread.start()
    assert client.started.wait(timeout=1.0)

    disconnect_thread = threading.Thread(target=lambda: session.set_infer(False))
    disconnect_thread.start()

    time.sleep(0.05)
    assert not client.disconnected.is_set()

    client.release.set()
    send_thread.join(timeout=1.0)
    disconnect_thread.join(timeout=1.0)

    assert client.disconnected.is_set()
    assert send_result["value"][0] == "imagezmq"
    assert session.client is None


def test_failed_connect_does_not_publish_inference_state(monkeypatch):
    class _Registry:
        @staticmethod
        def get(_name):
            return _FailingTransport

    session = TransportSession()
    monkeypatch.setattr("inference_streaming_benchmark.client.state.registry", _Registry)

    with pytest.raises(RuntimeError, match="connect failed"):
        session.set("zmq", True, 5555)

    assert session.snapshot() == (None, None, False)


def test_failed_transport_switch_keeps_existing_client(monkeypatch):
    class _Registry:
        @staticmethod
        def get(_name):
            return _FailingTransport

    session = TransportSession()
    client = _BlockingClient()
    session.client = client
    session.active_transport = "http_multipart"
    session.infer = True
    monkeypatch.setattr("inference_streaming_benchmark.client.state.registry", _Registry)

    with pytest.raises(RuntimeError, match="connect failed"):
        session.set("zmq", True, 5555)

    assert session.snapshot() == (client, "http_multipart", True)
    assert not client.disconnected.is_set()
