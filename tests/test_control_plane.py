import time
from unittest.mock import patch

import pytest

pytest.importorskip("fastapi")

import requests  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark import transports  # noqa: F401, E402 — triggers registration
from inference_streaming_benchmark.client_registry import ClientRegistry  # noqa: E402
from inference_streaming_benchmark.server import MultiRunManager, build_control_app  # noqa: E402
from inference_streaming_benchmark.transports import registry  # noqa: E402


class _FakeBatcher:
    """Stand-in for Batcher in control-plane tests. Stores state, no worker thread."""

    def __init__(self):
        self._state = {"enabled": False, "max_batch_size": 8, "max_wait_ms": 10.0}

    def state(self):
        return dict(self._state)

    def configure(self, *, enabled=None, max_batch_size=None, max_wait_ms=None):
        if max_batch_size is not None and max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_ms is not None and max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if enabled is not None:
            self._state["enabled"] = bool(enabled)
        if max_batch_size is not None:
            self._state["max_batch_size"] = int(max_batch_size)
        if max_wait_ms is not None:
            self._state["max_wait_ms"] = float(max_wait_ms)
        return dict(self._state)


class _FakeServer:
    """Stands in for Server — no sockets, no threads, just records interactions."""

    def __init__(self):
        self.active = None
        self.switch_calls: list[str] = []
        self.stop_called = False
        self.fail_on: str | None = None
        self.batcher = _FakeBatcher()

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


# ---- client registry endpoints ----


def test_register_and_clients_round_trip():
    reg = ClientRegistry()
    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        r = client.post(
            "/register",
            json={"name": "rpi-1", "ui_url": "http://10.0.0.5:8501", "version": "0.1"},
        )
        assert r.status_code == 200
        assert r.json()["ok"] is True

        r = client.get("/clients")
        assert r.status_code == 200
        payload = r.json()
        names = [c["name"] for c in payload["clients"]]
        assert names == ["rpi-1"]
        assert payload["clients"][0]["ui_url"] == "http://10.0.0.5:8501"


def test_heartbeat_updates_stats():
    reg = ClientRegistry()
    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        client.post("/register", json={"name": "rpi-1", "ui_url": "http://x:8501", "version": ""})
        r = client.post("/heartbeat", json={"name": "rpi-1", "stats": {"fps": 28.4, "backend": "grpc"}})
        assert r.status_code == 200

        r = client.get("/clients")
        clients = r.json()["clients"]
        assert clients[0]["stats"] == {"fps": 28.4, "backend": "grpc"}


def test_heartbeat_unknown_client_returns_404():
    with TestClient(build_control_app(_FakeServer())) as client:
        r = client.post("/heartbeat", json={"name": "ghost", "stats": {}})
    assert r.status_code == 404


def test_client_control_proxies_to_client_url():
    """POST /clients/{name}/control must forward the body to the client's /api/control."""
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")

    sent = {}

    class _FakeResponse:
        def __init__(self, body):
            self._body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self._body

    def _fake_post(url, json=None, timeout=None):
        sent["url"] = url
        sent["json"] = json
        return _FakeResponse(
            {
                "ok": True,
                "backend": json.get("backend"),
                "mock_camera": json.get("mock_camera"),
                "mock_delay_ms": json.get("mock_delay_ms"),
            }
        )

    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post", side_effect=_fake_post):
            r = client.post("/clients/rpi-1/control", json={"mock_camera": True})
            assert r.status_code == 200
            assert r.json()["mock_camera"] is True

    assert sent["url"] == "http://10.0.0.5:8501/api/control"
    # Optional fields with None must be stripped before forwarding.
    assert sent["json"] == {"mock_camera": True}

    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post", side_effect=_fake_post):
            r = client.post("/clients/rpi-1/control", json={"mock_delay_ms": 750})
            assert r.status_code == 200
            assert r.json()["mock_delay_ms"] == 750

    assert sent["json"] == {"mock_delay_ms": 750}


def test_client_control_unknown_client_returns_404():
    with TestClient(build_control_app(_FakeServer())) as client:
        r = client.post("/clients/ghost/control", json={"mock_camera": True})
    assert r.status_code == 404


def test_client_control_empty_body_is_400():
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")
    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        r = client.post("/clients/rpi-1/control", json={})
    assert r.status_code == 400


def test_clients_control_all_proxies_to_each_client_and_summarizes():
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")
    reg.register("rpi-2", "http://10.0.0.6:8501", "")
    reg.heartbeat("rpi-1", {})
    reg.heartbeat("rpi-2", {})
    server = _FakeServer()

    posted: list[tuple[str, dict]] = []

    class _OK:
        def raise_for_status(self):
            pass

    def _fake_post(url, json=None, **_kwargs):
        posted.append((url, json))
        if "10.0.0.6" in url:
            raise requests.RequestException("simulated unreachable")
        return _OK()

    with TestClient(build_control_app(server, reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post", side_effect=_fake_post):
            r = client.post("/clients/control-all", json={"backend": "grpc", "inference": True})

    assert r.status_code == 200
    assert server.switch_calls == ["grpc"]
    assert r.json()["results"] == {"rpi-1": "ok", "rpi-2": "failed"}
    assert sorted(url for url, _body in posted) == [
        "http://10.0.0.5:8501/api/control",
        "http://10.0.0.6:8501/api/control",
    ]
    assert all(body == {"backend": "grpc", "inference": True} for _url, body in posted)


def test_clients_control_all_strips_none_and_rejects_empty_body():
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")
    reg.heartbeat("rpi-1", {})
    sent = {}

    class _OK:
        def raise_for_status(self):
            pass

    def _fake_post(_url, json=None, **_kwargs):
        sent["json"] = json
        return _OK()

    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post", side_effect=_fake_post):
            r = client.post("/clients/control-all", json={"mock_delay_ms": 250})
            assert r.status_code == 200
            assert sent["json"] == {"mock_delay_ms": 250}

            r = client.post("/clients/control-all", json={})
            assert r.status_code == 400


def test_clients_control_all_rejects_unknown_backend():
    with TestClient(build_control_app(_FakeServer())) as client:
        r = client.post("/clients/control-all", json={"backend": "missing", "inference": True})
    assert r.status_code == 400


def test_switch_with_cascade_calls_each_client():
    """cascade=true forwards a /api/control POST to every registered client."""
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")
    reg.register("rpi-2", "http://10.0.0.6:8501", "")
    reg.heartbeat("rpi-1", {"inference": True})
    reg.heartbeat("rpi-2", {"inference": False})

    posted: list[tuple[str, dict]] = []

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {}

    def _fake_post(url, json=None, timeout=None):
        posted.append((url, json))
        return _Resp()

    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post", side_effect=_fake_post):
            r = client.post("/switch", json={"name": "grpc", "cascade": True})

    assert r.status_code == 200
    urls = sorted(u for u, _ in posted)
    assert urls == ["http://10.0.0.5:8501/api/control", "http://10.0.0.6:8501/api/control"]
    # Cascade preserves each client's last-known inference flag.
    by_url = {u: body for u, body in posted}
    assert by_url["http://10.0.0.5:8501/api/control"] == {"backend": "grpc", "inference": True}
    assert by_url["http://10.0.0.6:8501/api/control"] == {"backend": "grpc", "inference": False}


def test_switch_without_cascade_does_not_post_to_clients():
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")
    reg.heartbeat("rpi-1", {"inference": True})

    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post") as posted:
            r = client.post("/switch", json={"name": "grpc"})  # no cascade flag
    assert r.status_code == 200
    posted.assert_not_called()


def test_multi_run_start_and_status():
    calls = []

    def _runner(plan, *, control_base, duration_s, warmup_s, on_run_complete):
        calls.append((plan, control_base, duration_s, warmup_s))
        run = {"index": 1, "config": {"transport": plan[0].transport}}
        on_run_complete(run)
        return {"runs": [run]}

    manager = MultiRunManager(control_base="http://control", runner=_runner)
    with TestClient(build_control_app(_FakeServer(), multi_run_manager=manager)) as client:
        r = client.post(
            "/multi-run/start",
            json={
                "transports": ["grpc"],
                "batch_modes": ["off", "on"],
                "batch_sizes": [2],
                "batch_waits_ms": [5],
                "duration_s": 0.1,
                "warmup_s": 0,
            },
        )
        assert r.status_code == 200
        for _ in range(20):
            status = client.get("/multi-run/status").json()
            if not status["running"]:
                break
            time.sleep(0.01)

    assert status["error"] is None
    assert len(status["plan"]) == 2
    assert status["result"]["runs"][0]["config"] == {"transport": "grpc"}
    assert calls[0][1:] == ("http://control", 0.1, 0.0)


def test_multi_run_rejects_invalid_config():
    with TestClient(build_control_app(_FakeServer())) as client:
        r = client.post("/multi-run/start", json={"transports": ["missing"]})
        assert r.status_code == 400

        r = client.post("/multi-run/start", json={"transports": ["grpc"], "batch_modes": ["maybe"]})
        assert r.status_code == 400


def test_client_clear_proxies_to_client():
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")

    sent = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def _fake_post(url, **_kwargs):
        sent["url"] = url
        return _Resp()

    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post", side_effect=_fake_post):
            r = client.post("/clients/rpi-1/clear")
    assert r.status_code == 200
    assert sent["url"] == "http://10.0.0.5:8501/api/clear"


def test_batching_get_returns_current_state():
    server = _FakeServer()
    server.batcher.configure(enabled=True, max_batch_size=12, max_wait_ms=7.5)
    with TestClient(build_control_app(server)) as client:
        r = client.get("/batching")
    assert r.status_code == 200
    assert r.json() == {"enabled": True, "max_batch_size": 12, "max_wait_ms": 7.5}


def test_batching_post_partial_update_persists():
    server = _FakeServer()
    with TestClient(build_control_app(server)) as client:
        r = client.post("/batching", json={"enabled": True, "max_batch_size": 16})
        assert r.status_code == 200
        assert r.json() == {"enabled": True, "max_batch_size": 16, "max_wait_ms": 10.0}
        # Subsequent partial-update only flips the wait window.
        r = client.post("/batching", json={"max_wait_ms": 25.0})
        assert r.json() == {"enabled": True, "max_batch_size": 16, "max_wait_ms": 25.0}


def test_batching_post_rejects_invalid_values():
    server = _FakeServer()
    with TestClient(build_control_app(server)) as client:
        r = client.post("/batching", json={"max_batch_size": 0})
    assert r.status_code == 400
    assert "max_batch_size" in r.json()["detail"]


def test_clear_all_proxies_to_each_client_and_summarizes():
    reg = ClientRegistry()
    reg.register("rpi-1", "http://10.0.0.5:8501", "")
    reg.register("rpi-2", "http://10.0.0.6:8501", "")
    reg.heartbeat("rpi-1", {})
    reg.heartbeat("rpi-2", {})

    class _OK:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    def _fake_post(url, **_kwargs):
        if "10.0.0.6" in url:
            raise requests.RequestException("simulated unreachable")
        return _OK()

    with TestClient(build_control_app(_FakeServer(), reg)) as client:
        with patch("inference_streaming_benchmark.server.requests.post", side_effect=_fake_post):
            r = client.post("/clients/clear-all")
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == {"rpi-1": "ok", "rpi-2": "failed"}
