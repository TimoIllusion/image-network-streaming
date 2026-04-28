from __future__ import annotations

import statistics
import threading

import cv2

from inference_streaming_benchmark.config import CONTROL_HOST, MOCK_DELAY_MAX_MS, MOCK_DELAY_MS
from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports import registry
from inference_streaming_benchmark.transports.base import Transport

from .camera import CAMERA_MODES, initial_mode_from_env, open_camera

# Columns we collect per frame and show as medians in the stats table.
# Order is left-to-right additive: enc + dec + comms = transmission (transport overhead),
# then + queue_wait + infer + post = total. post_ms is server-side JSON serialization of
# detections (not transport). queue_wait_ms is server-side dynamic-batching queue wait
# (zero when batching is disabled or pass-through).
TIMING_COLUMNS = (
    "encode_ms",
    "decode_ms",
    "comms_ms",
    "transmission_ms",
    "queue_wait_ms",
    "infer_ms",
    "post_ms",
    "total_ms",
)
# Non-time numeric columns. Treated separately from TIMING_COLUMNS so they're not
# labeled "(ms)" and don't participate in the additive math.
NUMERIC_COLUMNS = ("batch_size",)


class CameraHandle:
    """Owns the cv2.VideoCapture (or fake) lifecycle. Supports runtime mode swap."""

    def __init__(self, initial_mode: str | None = None):
        self.cap: cv2.VideoCapture | None = None
        self.mode = initial_mode if initial_mode is not None else initial_mode_from_env()
        self._mock_delay_ms = self._clamp_mock_delay(MOCK_DELAY_MS)
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

    @staticmethod
    def _clamp_mock_delay(value: float) -> float:
        return min(MOCK_DELAY_MAX_MS, max(0.0, float(value)))

    def set_mock_delay_ms(self, value: float) -> None:
        with self._lock:
            self._mock_delay_ms = self._clamp_mock_delay(value)
        logger.info(f"mock camera inference delay max set: {self._mock_delay_ms:.1f}ms")

    def mock_delay_ms(self) -> float:
        with self._lock:
            return self._mock_delay_ms


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
            if not infer:
                self.infer = False
                self._disconnect_locked()
                return

            if self.active_transport == transport_name and self.client is not None:
                self.infer = True
                return

            cls = registry.get(transport_name)
            client = cls()
            try:
                client.connect(CONTROL_HOST, port)
            except Exception:
                if hasattr(client, "disconnect"):
                    try:
                        client.disconnect()
                    except Exception:
                        logger.exception("failed to clean up transport after connect failure")
                raise
            self._disconnect_locked()
            self.client = client
            self.active_transport = transport_name
            self.infer = True
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

    def send(self, frame, *, client_name: str, request_id: str):
        """Send one inference request on the active client without racing disconnect().

        Some transports, notably ImageZMQ/ZMQ, do not tolerate closing their socket
        from another thread while a REQ/REP send is in progress. Holding the session
        lock around send means control-plane switches wait for the in-flight request
        to finish or hit the transport timeout before disconnecting the old client.
        """
        with self.lock:
            if not self.infer or self.client is None or self.active_transport is None:
                return None, None, {}
            transport_name = self.active_transport
            detections, timings = self.client.send(frame, client_name=client_name, request_id=request_id)
            return transport_name, detections, timings


class BenchmarkCollector:
    """Thread-safe per-backend timing accumulator."""

    def __init__(self):
        self.bench_results: dict = {}
        self._lock = threading.RLock()

    def _bucket_key(self, transport_name: str, timings: dict) -> tuple:
        enabled = bool(timings.get("batching_enabled", False))
        max_batch_size = int(timings.get("batching_max_batch_size", 1))
        max_wait_ms = float(timings.get("batching_max_wait_ms", 0.0))
        inference_mode = str(timings.get("inference_mode", "unknown"))
        inference_instances = int(timings.get("inference_instances", 1))
        return (transport_name, enabled, max_batch_size, max_wait_ms, inference_mode, inference_instances)

    def _bucket_label(self, key: tuple) -> str:
        transport_name, enabled, max_batch_size, max_wait_ms, inference_mode, inference_instances = key
        if not enabled:
            batch_label = "batch off"
        else:
            batch_label = f"batch on size {max_batch_size} wait {max_wait_ms:g}ms"
        if inference_mode == "multi-instance":
            infer_label = f"infer {inference_mode} x{inference_instances}"
        else:
            infer_label = f"infer {inference_mode}"
        return f"{transport_name} · {batch_label} · {infer_label}"

    def record(self, transport_name: str, timings: dict) -> None:
        with self._lock:
            infer = timings.get("infer_ms", 0.0)
            post = timings.get("post_ms", 0.0)
            queue_wait = timings.get("queue_wait_ms", 0.0)
            encode = timings.get("encode_ms", 0.0)
            decode = timings.get("decode_ms", 0.0)
            total = timings.get("total_ms", 0.0)

            # `transmission` (network + codec only) is total minus everything that's
            # server-side processing or queueing.
            comms_ms = max(0.0, total - encode - decode - infer - post - queue_wait)
            transmission_ms = max(0.0, total - infer - post - queue_wait)
            timings = {**timings, "comms_ms": comms_ms, "transmission_ms": transmission_ms}

            key = self._bucket_key(transport_name, timings)
            bench = self.bench_results.setdefault(
                key,
                {
                    "transport_name": transport_name,
                    "bucket_label": self._bucket_label(key),
                    "batching_enabled": bool(key[1]),
                    "batching_max_batch_size": int(key[2]),
                    "batching_max_wait_ms": float(key[3]),
                    "inference_mode": key[4],
                    "inference_instances": int(key[5]),
                    "active_time_s": 0.0,
                    **{col: [] for col in TIMING_COLUMNS + NUMERIC_COLUMNS},
                },
            )
            for col in TIMING_COLUMNS + NUMERIC_COLUMNS:
                if col in timings:
                    bench[col].append(timings[col])
            bench["active_time_s"] += total / 1000

    def clear(self) -> None:
        with self._lock:
            self.bench_results = {}

    def build_stats_rows(self) -> list[dict]:
        with self._lock:
            rows = []
            for _key, data in self.bench_results.items():
                totals = data["total_ms"]
                if not totals:
                    continue
                duration_s = data["active_time_s"]
                row = {
                    "Backend": data.get("transport_name", "unknown"),
                    "Batch config": data.get("bucket_label", "unknown"),
                    "Frames": len(totals),
                    "Duration (s)": f"{duration_s:.1f}",
                    "FPS": f"{len(totals) / duration_s:.1f}" if duration_s > 0 else "-",
                }
                abbrevs = {
                    "transmission_ms": "transport (ms)",
                    "encode_ms": "enc (ms)",
                    "decode_ms": "dec (ms)",
                    "queue_wait_ms": "wait (ms)",
                }
                for col in TIMING_COLUMNS:
                    samples = list(data[col])
                    label = abbrevs.get(col, col.replace("_ms", " (ms)"))
                    row[label] = f"{statistics.median(samples):.1f}" if samples else "-"
                for col in NUMERIC_COLUMNS:
                    samples = list(data[col])
                    label = "batch" if col == "batch_size" else col
                    row[label] = f"{statistics.median(samples):.1f}" if samples else "-"
                rows.append(row)
            return rows

    def snapshot_for_heartbeat(self, active_transport: str | None) -> dict:
        """Heartbeat payload: currently-active backend + full per-backend history.

        `bench_rows` is the same shape as `build_stats_rows()` so the central UI can
        render a per-client breakdown table identical to the per-device stats table.
        Carrying all backends each heartbeat means the central UI never loses data
        when the operator switches transports.
        """
        return {"backend": active_transport, "bench_rows": self.build_stats_rows()}
