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
