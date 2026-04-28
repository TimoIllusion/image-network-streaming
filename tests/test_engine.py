import threading
import time

from inference_streaming_benchmark.engine import InferenceEngine


class _Result:
    def to_json(self):
        return "[]"


class _Model:
    def __init__(self):
        self.calls = []

    def __call__(self, images, **kwargs):
        self.calls.append((images, kwargs))
        if isinstance(images, list):
            return [_Result() for _ in images]
        return [_Result()]


class _BlockingModel:
    def __init__(self):
        self.active_calls = 0
        self.max_active_calls = 0
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, images, **kwargs):
        with self._lock:
            self.calls += 1
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        time.sleep(0.02)
        with self._lock:
            self.active_calls -= 1
        if isinstance(images, list):
            return [_Result() for _ in images]
        return [_Result()]


def test_infer_silences_ultralytics_verbose_output(monkeypatch):
    engine = InferenceEngine()
    model = _Model()
    monkeypatch.setattr(engine, "_get_or_load_model", lambda: model)

    engine.infer("image")

    assert model.calls[0][1]["verbose"] is False


def test_infer_batch_silences_ultralytics_verbose_output(monkeypatch):
    engine = InferenceEngine()
    model = _Model()
    monkeypatch.setattr(engine, "_get_or_load_model", lambda: model)

    engine.infer_batch(["image-a", "image-b"])

    assert model.calls[0][1]["verbose"] is False


def test_infer_serializes_model_calls(monkeypatch):
    engine = InferenceEngine()
    model = _BlockingModel()
    monkeypatch.setattr(engine, "_get_or_load_model", lambda: model)

    threads = [threading.Thread(target=lambda: engine.infer("image")) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1.0)

    assert model.calls == 4
    assert model.max_active_calls == 1
