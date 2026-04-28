import numpy as np

from inference_streaming_benchmark.transports import codec


def test_raw_codec_round_trips_shape_and_dtype():
    frame = np.arange(5 * 7 * 1, dtype=np.uint16).reshape((5, 7, 1))

    payload = codec.encode(frame, raw=True)
    decoded = codec.decode(payload, raw=True)

    assert decoded.shape == frame.shape
    assert decoded.dtype == frame.dtype
    np.testing.assert_array_equal(decoded, frame)


def test_raw_codec_decodes_legacy_shape_less_payload(monkeypatch):
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(codec, "FRAME_SHAPE", frame.shape)

    decoded = codec.decode(frame.tobytes(), raw=True)

    assert decoded.shape == frame.shape
