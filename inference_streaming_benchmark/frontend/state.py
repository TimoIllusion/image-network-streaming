from __future__ import annotations

import statistics
import threading

import cv2

from inference_streaming_benchmark.config import CONTROL_HOST
from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports import registry
from inference_streaming_benchmark.transports.base import Transport

from .camera import _open_camera

# Columns we collect per frame and show as medians in the stats table.
# transmission_ms = total - infer: end-to-end cost excluding only AI inference.
TIMING_COLUMNS = ("encode_ms", "decode_ms", "infer_ms", "post_ms", "comms_ms", "transmission_ms", "total_ms")


class CameraHandle:
    """Owns the cv2.VideoCapture (or fake) lifecycle. Initialized on first frame."""

    def __init__(self):
        self.cap: cv2.VideoCapture | None = None

    def ensure(self) -> cv2.VideoCapture:
        if self.cap is None:
            logger.info("Start camera initialization")
            self.cap = _open_camera()
            logger.info("Camera initialized")
        return self.cap


class TransportSession:
    """Owns the active transport client. Lock guards swaps so the MJPEG loop never sees a half-closed client."""

    def __init__(self):
        self.client: Transport | None = None
        self.active_transport: str | None = None
        self.infer: bool = False
        self.lock = threading.Lock()

    def set(self, transport_name: str, infer: bool, port: int) -> None:
        """Swap to (transport_name, port). Caller resolved the port via control_client."""
        with self.lock:
            self.infer = infer
            if not infer:
                self._disconnect_locked()
                return

            if self.active_transport == transport_name and self.client is not None:
                return

            self._disconnect_locked()
            cls = registry.get(transport_name)
            client = cls()
            client.connect(CONTROL_HOST, port)
            self.client = client
            self.active_transport = transport_name
            logger.info(f"client connected: {transport_name} → :{port}")

    def _disconnect_locked(self) -> None:
        if self.client is not None:
            try:
                self.client.disconnect()
            except Exception:
                logger.exception("failed to disconnect current client")
            self.client = None
            self.active_transport = None

    def snapshot(self) -> tuple[Transport | None, str | None, bool]:
        """Return (client, active_transport, infer) atomically — for the MJPEG read loop."""
        with self.lock:
            return self.client, self.active_transport, self.infer


class BenchmarkCollector:
    """Per-backend timing accumulator. Pure: no locks, no IO."""

    def __init__(self):
        self.bench_results: dict = {}

    def record(self, transport_name: str, timings: dict) -> None:
        server_ms = timings.get("decode_ms", 0.0) + timings.get("infer_ms", 0.0) + timings.get("post_ms", 0.0)
        comms_ms = max(0.0, timings.get("total_ms", 0.0) - timings.get("encode_ms", 0.0) - server_ms)
        transmission_ms = max(0.0, timings.get("total_ms", 0.0) - timings.get("infer_ms", 0.0))
        timings = {**timings, "comms_ms": comms_ms, "transmission_ms": transmission_ms}

        bench = self.bench_results.setdefault(transport_name, {"active_time_s": 0.0, **{col: [] for col in TIMING_COLUMNS}})
        for col in TIMING_COLUMNS:
            if col in timings:
                bench[col].append(timings[col])
        bench["active_time_s"] += timings.get("total_ms", 0.0) / 1000

    def clear(self) -> None:
        self.bench_results = {}

    def build_stats_rows(self) -> list[dict]:
        rows = []
        for name, data in self.bench_results.items():
            totals = data["total_ms"]
            if not totals:
                continue
            duration_s = data["active_time_s"]
            row = {
                "Backend": name,
                "Frames": len(totals),
                "Duration (s)": f"{duration_s:.1f}",
                "FPS": f"{len(totals) / duration_s:.1f}" if duration_s > 0 else "-",
            }
            for col in TIMING_COLUMNS:
                samples = data[col]
                row[col.replace("_ms", " (ms)")] = f"{statistics.median(samples):.1f}" if samples else "-"
            rows.append(row)
        return rows
