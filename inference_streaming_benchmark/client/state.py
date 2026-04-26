from __future__ import annotations

import statistics
import threading

import cv2

from inference_streaming_benchmark.config import CONTROL_HOST
from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports import registry
from inference_streaming_benchmark.transports.base import Transport

from .camera import CAMERA_MODES, initial_mode_from_env, open_camera

# Columns we collect per frame and show as medians in the stats table.
# Order is left-to-right additive: enc + dec + comms = transmission (transport overhead),
# then + infer + post = total. post_ms is server-side JSON serialization of detections, not transport.
TIMING_COLUMNS = ("encode_ms", "decode_ms", "comms_ms", "transmission_ms", "infer_ms", "post_ms", "total_ms")


class CameraHandle:
    """Owns the cv2.VideoCapture (or fake) lifecycle. Supports runtime mode swap."""

    def __init__(self, initial_mode: str | None = None):
        self.cap: cv2.VideoCapture | None = None
        self.mode = initial_mode if initial_mode is not None else initial_mode_from_env()
        self._lock = threading.Lock()

    def ensure(self) -> cv2.VideoCapture:
        with self._lock:
            if self.cap is None:
                self.cap = open_camera(self.mode)
            return self.cap

    def set_mode(self, mode: str) -> None:
        if mode not in CAMERA_MODES:
            raise ValueError(f"unknown camera mode: {mode!r}")
        with self._lock:
            if self.mode == mode and self.cap is not None:
                return
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    logger.exception("camera release failed")
                self.cap = None
            self.mode = mode
        logger.info(f"camera mode set: {mode}")


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

    def set_infer(self, infer: bool) -> None:
        """Toggle inference without changing transport. Used when only `inference` flips."""
        with self.lock:
            self.infer = infer
            if not infer:
                self._disconnect_locked()

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
        transmission_ms = max(0.0, timings.get("total_ms", 0.0) - timings.get("infer_ms", 0.0) - timings.get("post_ms", 0.0))
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
                abbrevs = {"transmission_ms": "transport (ms)", "encode_ms": "enc (ms)", "decode_ms": "dec (ms)"}
                label = abbrevs.get(col, col.replace("_ms", " (ms)"))
                row[label] = f"{statistics.median(samples):.1f}" if samples else "-"
            rows.append(row)
        return rows

    def snapshot_for_heartbeat(self, active_transport: str | None) -> dict:
        """Compact per-backend stats for the central UI's clients table."""
        if active_transport is None or active_transport not in self.bench_results:
            return {}
        data = self.bench_results[active_transport]
        totals = data["total_ms"]
        if not totals:
            return {"backend": active_transport}
        duration_s = data["active_time_s"]
        out = {
            "backend": active_transport,
            "frames": len(totals),
            "fps": (len(totals) / duration_s) if duration_s > 0 else 0.0,
        }
        for col in ("total_ms", "infer_ms", "transmission_ms"):
            samples = data[col]
            if samples:
                out[col] = float(statistics.median(samples))
        return out
