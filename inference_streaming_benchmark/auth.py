"""Opt-in shared-secret bearer auth for the control plane and per-device UI.

Defaults are LAN-friendly (no enforcement). When `INFSB_TOKEN` is set, every
non-loopback request must carry `Authorization: Bearer <token>` or it is
rejected with 401. Loopback (127.0.0.1, ::1) always passes through so local
curl/test/dev workflows stay simple.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def _is_loopback(request: Request) -> bool:
    client = request.client
    if client is None:
        return True
    return client.host in _LOOPBACK_HOSTS


def _expected_header(token: str) -> str:
    return f"Bearer {token}"


class BearerTokenMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, token: str):
        super().__init__(app)
        self._expected = _expected_header(token)

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        if _is_loopback(request):
            return await call_next(request)
        if request.headers.get("authorization") != self._expected:
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
        return await call_next(request)


def install_bearer_auth(app: FastAPI, token: str | None) -> None:
    """Register bearer auth middleware on `app`. No-op when `token` is None."""
    if not token:
        return
    app.add_middleware(BearerTokenMiddleware, token=token)


def outbound_headers(token: str | None) -> dict[str, str]:
    """Build the Authorization header dict for outbound calls. Empty when no token."""
    if not token:
        return {}
    return {"Authorization": _expected_header(token)}
