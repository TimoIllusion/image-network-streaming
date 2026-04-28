from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace

DEFAULT_STALE_AFTER_S = 10.0


@dataclass
class ClientRecord:
    name: str
    ui_url: str
    version: str
    registered_at: float
    last_heartbeat_at: float
    stats: dict = field(default_factory=dict)

    def snapshot(self) -> ClientRecord:
        return replace(self, stats=dict(self.stats))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ui_url": self.ui_url,
            "version": self.version,
            "registered_at": self.registered_at,
            "last_heartbeat_at": self.last_heartbeat_at,
            "age_s": max(0.0, time.time() - self.last_heartbeat_at),
            "stats": self.stats,
        }


class ClientRegistry:
    """In-memory registry of clients connected to this server.

    Thread-safe. No persistence — clients re-register on restart.
    """

    def __init__(self):
        self._records: dict[str, ClientRecord] = {}
        self._lock = threading.Lock()

    def register(self, name: str, ui_url: str, version: str) -> ClientRecord:
        now = time.time()
        record = ClientRecord(
            name=name,
            ui_url=ui_url,
            version=version,
            registered_at=now,
            last_heartbeat_at=now,
        )
        with self._lock:
            self._records[name] = record
            return record.snapshot()

    def heartbeat(self, name: str, stats: dict) -> ClientRecord:
        with self._lock:
            record = self._records.get(name)
            if record is None:
                raise KeyError(name)
            record.last_heartbeat_at = time.time()
            record.stats = stats
            return record.snapshot()

    def get(self, name: str) -> ClientRecord | None:
        with self._lock:
            record = self._records.get(name)
            return None if record is None else record.snapshot()

    def list_active(self, stale_after_s: float = DEFAULT_STALE_AFTER_S) -> list[ClientRecord]:
        cutoff = time.time() - stale_after_s
        with self._lock:
            stale = [name for name, r in self._records.items() if r.last_heartbeat_at < cutoff]
            for name in stale:
                del self._records[name]
            return [record.snapshot() for record in self._records.values()]

    def remove(self, name: str) -> bool:
        with self._lock:
            return self._records.pop(name, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._records.clear()
