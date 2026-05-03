from __future__ import annotations

import socket
import threading
from collections.abc import Callable

import requests

from inference_streaming_benchmark.auth import outbound_headers
from inference_streaming_benchmark.config import CONTROL_TIMEOUT_S, CONTROL_TOKEN
from inference_streaming_benchmark.logging import logger

HEARTBEAT_INTERVAL_S = 1.0


def get_local_ip(target_host: str) -> str:
    """Return the source IP this host would use to reach `target_host`.

    Uses a UDP socket connect (which doesn't actually send anything) to ask the
    kernel which local interface would route to that host. Falls back to
    hostname-based lookup, then loopback.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target_host, 1))
        return s.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            return "127.0.0.1"
    finally:
        s.close()


class Registrar:
    """Background thread that registers this client with the server and heartbeats every second.

    The heartbeat carries fresh stats from `stats_provider`, which the central UI reads
    to render its clients table. On HTTP errors, marks itself unregistered and re-registers
    on the next tick.
    """

    def __init__(
        self,
        name: str,
        control_base: str,
        ui_url: str,
        version: str,
        stats_provider: Callable[[], dict],
    ):
        self.name = name
        self.control_base = control_base
        self.ui_url = ui_url
        self.version = version
        self.stats_provider = stats_provider
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="client-registrar", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        registered = False
        while not self._stop.is_set():
            try:
                headers = outbound_headers(CONTROL_TOKEN)
                if not registered:
                    requests.post(
                        f"{self.control_base}/register",
                        json={"name": self.name, "ui_url": self.ui_url, "version": self.version},
                        headers=headers,
                        timeout=CONTROL_TIMEOUT_S,
                    ).raise_for_status()
                    logger.info(f"registered with server as {self.name!r} ({self.ui_url})")
                    registered = True
                requests.post(
                    f"{self.control_base}/heartbeat",
                    json={"name": self.name, "stats": self.stats_provider()},
                    headers=headers,
                    timeout=CONTROL_TIMEOUT_S,
                ).raise_for_status()
            except requests.HTTPError as e:
                # Server may have pruned us — re-register on next tick.
                if e.response is not None and e.response.status_code == 404:
                    registered = False
                else:
                    logger.warning(f"registrar HTTP {e.response.status_code if e.response else '?'} from server: {e}")
                    registered = False
            except requests.RequestException as e:
                logger.warning(f"registrar request failed: {e}")
                registered = False
            self._stop.wait(HEARTBEAT_INTERVAL_S)
