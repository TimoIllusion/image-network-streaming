from __future__ import annotations

import threading

import numpy as np

from inference_streaming_benchmark.config import CLIENT_NAME
from inference_streaming_benchmark.logging import logger

from .media import draw_detections, draw_fps
from .state import BenchmarkCollector, CameraHandle, TransportSession

IDLE_POLL_S = 0.2
ERROR_BACKOFF_S = 0.05


class FrameProcessor:
    """Background thread that captures frames and runs inference when enabled.

    Decouples inference from the MJPEG stream — inference runs whenever the user
    has it enabled, independent of whether a browser is viewing the per-device UI.
    The MJPEG generator becomes a passive observer of the latest annotated frame.

    Capture only runs when inference is on OR an MJPEG viewer is connected — so
    starting the client doesn't immediately open the webcam.
    """

    def __init__(self, camera: CameraHandle, session: TransportSession, collector: BenchmarkCollector):
        self.camera = camera
        self.session = session
        self.collector = collector
        self._latest: np.ndarray | None = None
        self._lock = threading.Lock()
        self._viewers = 0
        self._request_seq = 0
        self._viewers_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def viewer_started(self) -> None:
        with self._viewers_lock:
            self._viewers += 1

    def viewer_stopped(self) -> None:
        with self._viewers_lock:
            self._viewers = max(0, self._viewers - 1)

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="frame-processor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _should_capture(self) -> bool:
        _, _, infer = self.session.snapshot()
        with self._viewers_lock:
            viewers = self._viewers
        return infer or viewers > 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            if not self._should_capture():
                self._stop.wait(IDLE_POLL_S)
                continue

            try:
                cap = self.camera.ensure()
                ret, frame = cap.read()
            except Exception:
                logger.exception("frame capture failed")
                self._stop.wait(ERROR_BACKOFF_S)
                continue

            if not ret:
                self._stop.wait(ERROR_BACKOFF_S)
                continue

            active_client, active_transport, infer = self.session.snapshot()

            if infer and active_client is not None and active_transport is not None:
                try:
                    self._request_seq += 1
                    request_id = f"{CLIENT_NAME}-{self._request_seq:06d}"
                    detections, timings = active_client.send(
                        frame,
                        client_name=CLIENT_NAME,
                        request_id=request_id,
                    )
                    self.collector.record(active_transport, timings)
                    fps = 1000 / timings["total_ms"] if timings.get("total_ms", 0) > 0 else 0
                    if detections is not None:
                        frame = draw_detections(frame, detections)
                        frame = draw_fps(frame, fps)
                except Exception:
                    logger.exception("inference send failed")

            with self._lock:
                self._latest = frame
