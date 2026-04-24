import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark import transports  # noqa: F401, E402 — triggers registration
from inference_streaming_benchmark.server import build_control_app  # noqa: E402
from inference_streaming_benchmark.transports import registry  # noqa: E402


class _FakeServer:
    """Stands in for Server — no sockets, no threads, just records interactions."""

    def __init__(self):
        self.active = None
        self.switch_calls: list[str] = []
        self.stop_called = False
        self.fail_on: str | None = None

    def switch(self, name: str):
        self.switch_calls.append(name)
        if self.fail_on == name:
            raise RuntimeError("simulated failure")
        cls = registry.get(name)
        # Bypass __init__ so we don't allocate ZMQ contexts / FastAPI apps etc.
        instance = object.__new__(cls)
        self.active = instance
        return instance

    def stop(self):
        self.stop_called = True


def test_health():
    with TestClient(build_control_app(_FakeServer())) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_transports_endpoint_lists_registry():
    with TestClient(build_control_app(_FakeServer())) as client:
        r = client.get("/transports")

    assert r.status_code == 200
    items = r.json()
    by_name = {item["name"]: item for item in items}
    assert set(by_name) == {
        "http_multipart",
        "http_multipart_raw",
        "zmq",
        "zmq_raw",
        "websocket",
        "websocket_raw",
        "imagezmq",
        "grpc",
    }
    # Spot-check port round-trip for the new entries.
    assert by_name["websocket"]["port"] == 8009
    assert by_name["websocket_raw"]["port"] == 8011
    assert by_name["http_multipart_raw"]["port"] == 8010
    assert by_name["zmq_raw"]["port"] == 5557
    # Every entry carries its display name.
    assert all(item["display_name"] for item in items)
    # Nothing is active when server.active is None.
    assert all(item["active"] is False for item in items)


def test_transports_endpoint_marks_active():
    server = _FakeServer()
    # Pretend http_multipart is already running.
    server.active = object.__new__(registry.get("http_multipart"))
    with TestClient(build_control_app(server)) as client:
        r = client.get("/transports")
    by_name = {item["name"]: item for item in r.json()}
    assert by_name["http_multipart"]["active"] is True
    # Exactly one is active.
    assert sum(1 for item in r.json() if item["active"]) == 1


def test_switch_rejects_unknown_transport():
    server = _FakeServer()
    with TestClient(build_control_app(server)) as client:
        r = client.post("/switch", json={"name": "no_such_transport"})
    assert r.status_code == 400
    assert "unknown transport" in r.json()["detail"]
    # Validation fired before delegating to the server.
    assert server.switch_calls == []


def test_switch_success_surfaces_name_and_port():
    server = _FakeServer()
    with TestClient(build_control_app(server)) as client:
        r = client.post("/switch", json={"name": "websocket_raw"})
    assert r.status_code == 200
    assert r.json() == {"name": "websocket_raw", "port": 8011, "active": True}
    assert server.switch_calls == ["websocket_raw"]


def test_switch_surfaces_500_on_transport_startup_failure():
    server = _FakeServer()
    server.fail_on = "zmq"
    with TestClient(build_control_app(server)) as client:
        r = client.post("/switch", json={"name": "zmq"})
    assert r.status_code == 500
    assert "simulated failure" in r.json()["detail"]


def test_lifespan_shutdown_calls_server_stop():
    server = _FakeServer()
    with TestClient(build_control_app(server)):
        pass  # leaving the context triggers lifespan shutdown
    assert server.stop_called is True
