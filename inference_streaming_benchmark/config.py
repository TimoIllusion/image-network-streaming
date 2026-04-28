from __future__ import annotations

import os
import socket

CONTROL_HOST = os.getenv("INFSB_CONTROL_HOST", "localhost")
CONTROL_PORT = int(os.getenv("INFSB_CONTROL_PORT", "9000"))
UI_PORT = int(os.getenv("INFSB_UI_PORT", "8501"))
CONTROL_TIMEOUT_S = float(os.getenv("INFSB_CONTROL_TIMEOUT_S", "5.0"))
CONTROL_BASE = f"http://{CONTROL_HOST}:{CONTROL_PORT}"
CLIENT_NAME = os.getenv("INFSB_CLIENT_NAME") or socket.gethostname()
MOCK_DELAY_MS = float(os.getenv("INFSB_MOCK_DELAY_MS", "100"))
MOCK_DELAY_MAX_MS = 5000.0

# Dynamic batching defaults (server-side). Runtime-tunable via the central UI.
BATCH_ENABLED = os.getenv("INFSB_BATCH_ENABLED", "0") == "1"
BATCH_SIZE = int(os.getenv("INFSB_BATCH_SIZE", "8"))
BATCH_WAIT_MS = float(os.getenv("INFSB_BATCH_WAIT_MS", "10"))
