from __future__ import annotations

import queue
import threading
import time
from concurrent.futures import Future

import numpy as np

from inference_streaming_benchmark.logging import logger
from inference_streaming_benchmark.transports.base import InferenceRequest


class Batcher:
    """Coalesces concurrent inference calls into batched model invocations.

    When ``enabled=False`` this is a literal pass-through to ``engine.infer``
    (zero overhead — the existing single-stream benchmark numbers stay valid).

    When ``enabled=True`` each ``infer(image)`` call drops ``(image, future)``
    into an internal queue. A worker thread pulls up to ``max_batch_size`` items
    waiting up to ``max_wait_ms``, calls ``engine.infer_batch(images)``, and
    fans the results back to each waiting caller.

    Per-frame timings gain two fields the existing ones don't have:
        - ``queue_wait_ms``: total time before the model call started
        - ``backlog_wait_ms``: time spent queued behind earlier inference work
        - ``batch_fill_wait_ms``: time spent waiting for more frames in this batch
        - ``batch_size``: how many frames went through the model together
    Pass-through mode reports zero wait and ``batch_size=1`` so the columns are
    always present in the stats.
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

    def infer(self, request):
        """Same shape as ``engine.infer`` — returns ``(detections, timings)``."""
        request = self._normalize_request(request)
        with self._cfg_lock:
            enabled = self._enabled
        if not enabled:
            t0 = time.perf_counter()
            detections, timings = self._engine.infer(request.image)
            full_timings = {
                **timings,
                "queue_wait_ms": 0.0,
                "backlog_wait_ms": 0.0,
                "batch_fill_wait_ms": 0.0,
                "batch_size": 1,
            }
            self._log_request(request, full_timings, mode="direct", total_server_ms=(time.perf_counter() - t0) * 1000)
            return detections, full_timings

        future: Future = Future()
        self._queue.put((request, future, time.perf_counter()))
        return future.result()

    def _normalize_request(self, request) -> InferenceRequest:
        if isinstance(request, InferenceRequest):
            return request
        if isinstance(request, np.ndarray):
            return InferenceRequest(image=request, received_at=time.perf_counter())
        return InferenceRequest(image=request, received_at=time.perf_counter())

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
            batch_opened_at = time.perf_counter()
            deadline = batch_opened_at + max_wait_ms / 1000.0
            while len(batch) < max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            self._dispatch(batch, batch_opened_at)

    def _dispatch(self, batch: list, batch_opened_at: float) -> None:
        t_call_start = time.perf_counter()
        try:
            requests = [item[0] for item in batch]
            results = self._engine.infer_batch([request.image for request in requests])
        except Exception as exc:
            for _request, future, _enqueued_at in batch:
                future.set_exception(exc)
            return

        if len(results) != len(batch):
            err = RuntimeError(f"infer_batch returned {len(results)} results for batch of {len(batch)}")
            for _request, future, _enqueued_at in batch:
                future.set_exception(err)
            return

        batch_size = len(batch)
        if batch_size > 0:
            max_infer_ms = max((timings.get("infer_ms", 0.0) for _detections, timings in results), default=0.0)
            clients = ",".join(request.client_name for request, _future, _enqueued_at in batch)
            waits = [self._split_wait(enqueued_at, batch_opened_at, t_call_start) for _request, _future, enqueued_at in batch]
            max_backlog_wait_ms = max((wait["backlog_wait_ms"] for wait in waits), default=0.0)
            max_batch_fill_wait_ms = max((wait["batch_fill_wait_ms"] for wait in waits), default=0.0)
            with self._cfg_lock:
                max_batch_size = self._max_batch_size
                max_wait_ms = self._max_wait_ms
            logger.info(
                f"batch dispatch size={batch_size} max_size={max_batch_size} "
                f"max_wait={max_wait_ms:.1f}ms backlog_wait_max={max_backlog_wait_ms:.1f}ms "
                f"batch_fill_wait_max={max_batch_fill_wait_ms:.1f}ms clients={clients} infer={max_infer_ms:.1f}ms"
            )
        for (request, future, enqueued_at), (detections, timings) in zip(batch, results, strict=True):
            waits = self._split_wait(enqueued_at, batch_opened_at, t_call_start)
            full_timings = {**timings, **waits, "batch_size": batch_size}
            self._log_request(request, full_timings, mode="batch")
            future.set_result((detections, full_timings))

    def _split_wait(self, enqueued_at: float, batch_opened_at: float, t_call_start: float) -> dict[str, float]:
        backlog_wait_ms = max(0.0, (batch_opened_at - enqueued_at) * 1000.0)
        batch_fill_start = max(enqueued_at, batch_opened_at)
        batch_fill_wait_ms = max(0.0, (t_call_start - batch_fill_start) * 1000.0)
        return {
            "queue_wait_ms": backlog_wait_ms + batch_fill_wait_ms,
            "backlog_wait_ms": backlog_wait_ms,
            "batch_fill_wait_ms": batch_fill_wait_ms,
        }

    def _log_request(
        self,
        request: InferenceRequest,
        timings: dict,
        *,
        mode: str,
        total_server_ms: float | None = None,
    ) -> None:
        parts = [
            f"infer request_id={request.request_id}",
            f"client={request.client_name}",
            f"transport={request.transport}",
            f"mode={mode}",
            f"batch_size={int(timings.get('batch_size', 1))}",
            f"queue_wait={timings.get('queue_wait_ms', 0.0):.1f}ms",
            f"backlog_wait={timings.get('backlog_wait_ms', 0.0):.1f}ms",
            f"batch_fill_wait={timings.get('batch_fill_wait_ms', 0.0):.1f}ms",
            f"infer={timings.get('infer_ms', 0.0):.1f}ms",
            f"post={timings.get('post_ms', 0.0):.1f}ms",
        ]
        if total_server_ms is not None:
            parts.append(f"total_server={total_server_ms:.1f}ms")
        logger.info(" ".join(parts))
