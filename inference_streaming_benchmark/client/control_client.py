from __future__ import annotations

import requests

from inference_streaming_benchmark.auth import outbound_headers
from inference_streaming_benchmark.config import CONTROL_BASE, CONTROL_TIMEOUT_S, CONTROL_TOKEN


def switch_transport(name: str) -> int:
    """Ask the AI server to switch transports. Returns the bound port."""
    resp = requests.post(
        f"{CONTROL_BASE}/switch",
        json={"name": name},
        headers=outbound_headers(CONTROL_TOKEN),
        timeout=CONTROL_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["port"]


def fetch_transports() -> list[dict]:
    """Returns the AI server's `/transports` payload (list of transport descriptors)."""
    r = requests.get(
        f"{CONTROL_BASE}/transports",
        headers=outbound_headers(CONTROL_TOKEN),
        timeout=CONTROL_TIMEOUT_S,
    )
    r.raise_for_status()
    return r.json()
