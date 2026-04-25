from __future__ import annotations

import os

CONTROL_HOST = os.getenv("INFSB_CONTROL_HOST", "localhost")
CONTROL_PORT = int(os.getenv("INFSB_CONTROL_PORT", "9000"))
UI_PORT = int(os.getenv("INFSB_UI_PORT", "8501"))
CONTROL_TIMEOUT_S = float(os.getenv("INFSB_CONTROL_TIMEOUT_S", "5.0"))
CONTROL_BASE = f"http://{CONTROL_HOST}:{CONTROL_PORT}"
