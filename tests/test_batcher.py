import threading
import time

import pytest

from inference_streaming_benchmark.batcher import Batcher


class _FakeEngine:
    """Stand-in for InferenceEngine — records calls, returns deterministic results."""

    def __init__(self, infer_ms: float = 5.0):
        self.infer_ms = infer_ms
        self.infer_calls = 0
        self.infer_batch_calls: list[int] = []
        self._lock = threading.Lock()

    def infer(self, image):
        with self._lock:
            self.infer_calls += 1
        return ([{"img": image}], {"infer_ms": self.infer_ms, "post_ms": 0.0})

    def infer_batch(self, images):
        with self._lock:
            self.infer_batch_calls.append(len(images))
        time.sleep(0.001)
        return [([{"img": img}], {"infer_ms": self.infer_ms, "post_ms": 0.1}) for img in images]


def test_passthrough_when_disabled_does_not_use_queue():
    engine = _FakeEngine()
    batcher = Batcher(engine, enabled=False)
    try:
        detections, timings = batcher.infer("img-A")
        assert detections == [{"img": "img-A"}]
        # Pass-through still annotates the timings so downstream columns are stable.
        assert timings["queue_wait_ms"] == 0.0
        assert timings["batch_size"] == 1
        # infer was used, not infer_batch.
        assert engine.infer_calls == 1
        assert engine.infer_batch_calls == []
    finally:
        batcher.stop()


def test_enabled_coalesces_concurrent_calls_into_one_batch():
    """N concurrent infer() calls within max_wait_ms must hit infer_batch once with all N images."""
    engine = _FakeEngine()
    batcher = Batcher(engine, enabled=True, max_batch_size=8, max_wait_ms=50.0)
    try:
        results = [None] * 4
        threads = []

        def call(idx):
            results[idx] = batcher.infer(f"img-{idx}")

        for i in range(4):
            t = threading.Thread(target=call, args=(i,))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert all(r is not None for r in results)
        # All four hit one batched model call (no spurious singleton calls).
        assert engine.infer_batch_calls == [4]
        # Each caller's frame round-trips its identity and learns the batch it was in.
        for i, (detections, timings) in enumerate(results):
            assert detections == [{"img": f"img-{i}"}]
            assert timings["batch_size"] == 4
            assert timings["queue_wait_ms"] >= 0.0
    finally:
        batcher.stop()


def test_max_batch_size_caps_batch():
    engine = _FakeEngine()
    batcher = Batcher(engine, enabled=True, max_batch_size=2, max_wait_ms=200.0)
    try:
        results = [None] * 5
        threads = []
        for i in range(5):
            t = threading.Thread(target=lambda i=i: results.__setitem__(i, batcher.infer(f"img-{i}")))
            threads.append(t)
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=3.0)
        # 5 frames, max_batch_size=2 → batches of size 2, 2, 1 (total of 5).
        assert sum(engine.infer_batch_calls) == 5
        assert max(engine.infer_batch_calls) == 2
    finally:
        batcher.stop()


def test_max_wait_ms_flushes_singleton_when_no_more_arrive():
    engine = _FakeEngine()
    batcher = Batcher(engine, enabled=True, max_batch_size=8, max_wait_ms=20.0)
    try:
        t0 = time.perf_counter()
        detections, timings = batcher.infer("only")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert detections == [{"img": "only"}]
        assert timings["batch_size"] == 1
        # Worker waited ~max_wait_ms before flushing; allow generous slack for CI jitter.
        assert elapsed_ms >= 15.0
        assert engine.infer_batch_calls == [1]
    finally:
        batcher.stop()


def test_configure_partial_update():
    engine = _FakeEngine()
    batcher = Batcher(engine, enabled=False, max_batch_size=4, max_wait_ms=10.0)
    try:
        batcher.configure(enabled=True, max_batch_size=16)
        s = batcher.state()
        assert s == {"enabled": True, "max_batch_size": 16, "max_wait_ms": 10.0}

        # Disabling at runtime falls back to pass-through immediately.
        batcher.configure(enabled=False)
        engine.infer_calls = 0
        engine.infer_batch_calls.clear()
        batcher.infer("after-disable")
        assert engine.infer_calls == 1
        assert engine.infer_batch_calls == []
    finally:
        batcher.stop()


def test_configure_rejects_bad_values():
    engine = _FakeEngine()
    batcher = Batcher(engine, enabled=False)
    try:
        with pytest.raises(ValueError):
            batcher.configure(max_batch_size=0)
        with pytest.raises(ValueError):
            batcher.configure(max_wait_ms=-1)
    finally:
        batcher.stop()


def test_engine_exception_propagates_to_each_caller():
    class _Boom(_FakeEngine):
        def infer_batch(self, images):
            raise RuntimeError("model exploded")

    engine = _Boom()
    batcher = Batcher(engine, enabled=True, max_batch_size=4, max_wait_ms=20.0)
    try:
        results = [None] * 3
        errors = [None] * 3
        threads = []

        def call(idx):
            try:
                results[idx] = batcher.infer(f"img-{idx}")
            except Exception as e:
                errors[idx] = e

        for i in range(3):
            threads.append(threading.Thread(target=call, args=(i,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=2.0)

        assert all(r is None for r in results)
        assert all(isinstance(e, RuntimeError) for e in errors)
    finally:
        batcher.stop()
