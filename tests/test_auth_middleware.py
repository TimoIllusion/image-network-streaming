import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from inference_streaming_benchmark.auth import install_bearer_auth, outbound_headers  # noqa: E402


def _build_app(token: str | None) -> FastAPI:
    app = FastAPI()
    install_bearer_auth(app, token)

    @app.get("/ping")
    def ping():
        return {"ok": True}

    return app


def _request(app: FastAPI, *, host: str, headers: dict | None = None):
    """TestClient defaults `client=("testclient", 50000)` — non-loopback. Override per-call."""
    with TestClient(app, client=(host, 12345)) as client:
        return client.get("/ping", headers=headers or {})


def test_no_token_lets_everything_through():
    app = _build_app(None)
    assert _request(app, host="10.0.0.1").status_code == 200
    assert _request(app, host="127.0.0.1").status_code == 200


def test_token_set_loopback_passes_without_header():
    app = _build_app("secret")
    assert _request(app, host="127.0.0.1").status_code == 200
    assert _request(app, host="::1").status_code == 200


def test_token_set_non_loopback_missing_header_is_401():
    app = _build_app("secret")
    r = _request(app, host="10.0.0.1")
    assert r.status_code == 401


def test_token_set_non_loopback_wrong_header_is_401():
    app = _build_app("secret")
    r = _request(app, host="10.0.0.1", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_token_set_non_loopback_correct_header_passes():
    app = _build_app("secret")
    r = _request(app, host="10.0.0.1", headers={"Authorization": "Bearer secret"})
    assert r.status_code == 200


def test_outbound_headers_empty_when_no_token():
    assert outbound_headers(None) == {}
    assert outbound_headers("") == {}


def test_outbound_headers_emits_bearer_when_token_set():
    assert outbound_headers("secret") == {"Authorization": "Bearer secret"}
