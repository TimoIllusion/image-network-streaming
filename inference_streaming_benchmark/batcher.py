from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future


class Batcher:
    """Coalesces concurrent inference calls into batched model invocations.

    When ``enabled=False`` this is a literal pass-through to ``engine.infer``
    (zero overhead — the existing single-stream benchmark numbers stay valid).

    When ``enabled=True`` each ``infer(image)`` call drops ``(image, future)``
    into an internal queue. A worker thread pulls up to ``max_batch_size`` items
    waiting up to ``max_wait_ms``, calls ``engine.infer_batch(images)``, and
    fans the results back to each waiting caller.

    Per-frame timings gain two fields the existing ones don't have:
        - ``queue_wait_ms``: time the frame sat in the queue before the model call started
        - ``batch_size``: how many frames went through the model together
    Pass-through mode reports ``queue_wait_ms=0.0`` and ``batch_size=1`` so the
    columns are always present in the stats.
    """

    def __init__(
        self,
        engine,
        *,
        enabled: bool = False,
        max_batch_size: int = 8,
        max_wait_ms: float = 10.0,
    ):
        self._engine = engine
        self._enabled = bool(enabled)
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_ms = max(0.0, float(max_wait_ms))
        self._queue: queue.Queue = queue.Queue()
        self._cfg_lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="batcher-worker")
        self._worker.start()

    def state(self) -> dict:
        with self._cfg_lock:
            return {
                "enabled": self._enabled,
                "max_batch_size": self._max_batch_size,
                "max_wait_ms": self._max_wait_ms,
            }

    def configure(
        self,
        *,
        enabled: bool | None = None,
        max_batch_size: int | None = None,
        max_wait_ms: float | None = None,
    ) -> dict:
        with self._cfg_lock:
            if max_batch_size is not None:
                if int(max_batch_size) < 1:
                    raise ValueError("max_batch_size must be >= 1")
                self._max_batch_size = int(max_batch_size)
            if max_wait_ms is not None:
                if float(max_wait_ms) < 0:
                    raise ValueError("max_wait_ms must be >= 0")
                self._max_wait_ms = float(max_wait_ms)
            if enabled is not None:
                self._enabled = bool(enabled)
            return {
                "enabled": self._enabled,
                "max_batch_size": self._max_batch_size,
                "max_wait_ms": self._max_wait_ms,
            }

    def stop(self) -> None:
        self._stop.set()
        if self._worker.is_alive():
            self._worker.join(timeout=2.0)

    def infer(self, image):
        """Same shape as ``engine.infer`` — returns ``(detections, timings)``."""
        with self._cfg_lock:
            enabled = self._enabled
        if not enabled:
            detections, timings = self._engine.infer(image)
            return detections, {**timings, "queue_wait_ms": 0.0, "batch_size": 1}

        future: Future = Future()
        self._queue.put((image, future, time.perf_counter()))
        return future.result()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            with self._cfg_lock:
                max_batch_size = self._max_batch_size
                max_wait_ms = self._max_wait_ms

            batch = [first]
            deadline = time.perf_counter() + max_wait_ms / 1000.0
            while len(batch) < max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            self._dispatch(batch)

    def _dispatch(self, batch: list) -> None:
        t_call_start = time.perf_counter()
        try:
            results = self._engine.infer_batch([item[0] for item in batch])
        except Exception as exc:
            for _image, future, _enqueued_at in batch:
                future.set_exception(exc)
            return

        if len(results) != len(batch):
            err = RuntimeError(f"infer_batch returned {len(results)} results for batch of {len(batch)}")
            for _image, future, _enqueued_at in batch:
                future.set_exception(err)
            return

        batch_size = len(batch)
        for (_image, future, enqueued_at), (detections, timings) in zip(batch, results, strict=True):
            queue_wait_ms = max(0.0, (t_call_start - enqueued_at) * 1000.0)
            future.set_result((detections, {**timings, "queue_wait_ms": queue_wait_ms, "batch_size": batch_size}))
