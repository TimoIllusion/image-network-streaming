from __future__ import annotations

import os
import socket

CONTROL_HOST = os.getenv("INFSB_CONTROL_HOST", "localhost")
CONTROL_PORT = int(os.getenv("INFSB_CONTROL_PORT", "9000"))
UI_PORT = int(os.getenv("INFSB_UI_PORT", "8501"))
CONTROL_TIMEOUT_S = float(os.getenv("INFSB_CONTROL_TIMEOUT_S", "5.0"))
CONTROL_BASE = f"http://{CONTROL_HOST}:{CONTROL_PORT}"
CONTROL_BIND = os.getenv("INFSB_BIND", "0.0.0.0")
CLIENT_NAME = os.getenv("INFSB_CLIENT_NAME") or socket.gethostname()
MOCK_DELAY_MS = float(os.getenv("INFSB_MOCK_DELAY_MS", "100"))
MOCK_DELAY_MAX_MS = 5000.0

_TRANSPORT_PORT_ENV_PREFIX = "INFSB_TRANSPORT_PORT_"
_TRANSPORT_DEFAULT_PORTS = {
    "http_multipart": 8008,
    "http_multipart_raw": 8010,
    "zmq": 5555,
    "zmq_raw": 5557,
    "websocket": 8009,
    "websocket_raw": 8011,
    "imagezmq": 5556,
    "grpc": 50051,
}


def _transport_port_env_name(name: str) -> str:
    return f"{_TRANSPORT_PORT_ENV_PREFIX}{name.upper()}"


def _transport_port(name: str, default: int) -> int:
    return int(os.getenv(_transport_port_env_name(name), str(default)))


TRANSPORT_DEFAULT_PORTS = {name: _transport_port(name, default) for name, default in _TRANSPORT_DEFAULT_PORTS.items()}

# Dynamic batching defaults (server-side). Runtime-tunable via the central UI.
BATCH_ENABLED = os.getenv("INFSB_BATCH_ENABLED", "0") == "1"
BATCH_SIZE = int(os.getenv("INFSB_BATCH_SIZE", "8"))
BATCH_WAIT_MS = float(os.getenv("INFSB_BATCH_WAIT_MS", "10"))

# Inference concurrency mode (server-side). Runtime-tunable via the central UI.
INFER_MODE = os.getenv("INFSB_INFER_MODE", "single")
INFER_INSTANCES = int(os.getenv("INFSB_INFER_INSTANCES", "2"))
