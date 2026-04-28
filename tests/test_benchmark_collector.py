import pytest

pytest.importorskip("cv2")
pytest.importorskip("fastapi")

from inference_streaming_benchmark.client.state import BenchmarkCollector  # noqa: E402


def _only_bench(collector):
    return next(iter(collector.bench_results.values()))


def test_record_derives_comms_and_transmission():
    """Comms = total − encode − (decode+infer+post); transmission = total − infer."""
    collector = BenchmarkCollector()
    collector.record(
        "zmq",
        {
            "encode_ms": 2.0,
            "decode_ms": 3.0,
            "infer_ms": 20.0,
            "post_ms": 1.0,
            "total_ms": 30.0,
        },
    )

    bench = _only_bench(collector)
    # server_ms = 3 + 20 + 1 = 24; comms = 30 − 2 − 24 = 4
    assert bench["comms_ms"] == [4.0]
    # transport = total − infer − post = 30 − 20 − 1 = 9 (excludes server-side JSON serialization)
    assert bench["transmission_ms"] == [9.0]
    # raw columns round-trip unchanged
    assert bench["encode_ms"] == [2.0]
    assert bench["total_ms"] == [30.0]
    # active_time_s accumulates total_ms / 1000
    assert bench["active_time_s"] == pytest.approx(0.030)


def test_record_clamps_negative_comms_to_zero():
    """Clock jitter or weird fake handlers can make encode+server exceed total;
    clamp avoids poisoning the median with negative numbers."""
    collector = BenchmarkCollector()
    collector.record(
        "zmq",
        {
            "encode_ms": 50.0,  # absurd: larger than total
            "decode_ms": 1.0,
            "infer_ms": 1.0,
            "post_ms": 1.0,
            "total_ms": 10.0,  # would yield comms = 10 − 50 − 3 = −43
        },
    )
    bench = _only_bench(collector)
    assert bench["comms_ms"] == [0.0]
    # transport = total − infer − post = 10 − 1 − 1 = 8 (not clamped: positive)
    assert bench["transmission_ms"] == [8.0]


def test_build_stats_rows_computes_median_fps_and_frame_count():
    collector = BenchmarkCollector()
    # Three frames on http_multipart with rising totals → median is the middle value.
    for total in (10.0, 20.0, 30.0):
        collector.record(
            "http_multipart",
            {"encode_ms": 1.0, "decode_ms": 1.0, "infer_ms": 5.0, "post_ms": 0.5, "total_ms": total},
        )
    # One frame on zmq so we exercise the multi-backend loop.
    collector.record(
        "zmq",
        {"encode_ms": 2.0, "decode_ms": 2.0, "infer_ms": 6.0, "post_ms": 1.0, "total_ms": 25.0},
    )

    rows = {r["Backend"]: r for r in collector.build_stats_rows()}

    http_row = rows["http_multipart"]
    assert http_row["Batch config"] == "http_multipart · batch off · infer unknown"
    assert http_row["Frames"] == 3
    # Median of [10, 20, 30] = 20
    assert http_row["total (ms)"] == "20.0"
    # Duration is sum/1000 = 0.060; FPS = 3 / 0.060 = 50
    assert http_row["Duration (s)"] == "0.1"
    assert http_row["FPS"] == "50.0"

    zmq_row = rows["zmq"]
    assert zmq_row["Frames"] == 1
    assert zmq_row["total (ms)"] == "25.0"


def test_same_backend_splits_rows_by_batch_config():
    collector = BenchmarkCollector()
    collector.record(
        "http_multipart",
        {
            "encode_ms": 1.0,
            "decode_ms": 1.0,
            "infer_ms": 5.0,
            "post_ms": 0.5,
            "total_ms": 20.0,
            "batching_enabled": False,
            "batching_max_batch_size": 1,
            "batching_max_wait_ms": 0.0,
        },
    )
    collector.record(
        "http_multipart",
        {
            "encode_ms": 1.0,
            "decode_ms": 1.0,
            "infer_ms": 5.0,
            "post_ms": 0.5,
            "total_ms": 30.0,
            "batching_enabled": True,
            "batching_max_batch_size": 3,
            "batching_max_wait_ms": 25.0,
        },
    )

    rows = sorted(collector.build_stats_rows(), key=lambda r: r["Batch config"])

    assert [r["Batch config"] for r in rows] == [
        "http_multipart · batch off · infer unknown",
        "http_multipart · batch on size 3 wait 25ms · infer unknown",
    ]
    assert [r["Frames"] for r in rows] == [1, 1]


def test_build_stats_rows_skips_backends_with_no_samples():
    """A backend that was selected but recorded zero frames should not appear as an all-dash row."""
    collector = BenchmarkCollector()
    # setdefault creates the empty buckets without appending anything
    collector.bench_results.setdefault(
        ("websocket", False, 1, 0.0, "unknown", 1),
        {
            "transport_name": "websocket",
            "bucket_label": "websocket · batch off",
            "active_time_s": 0.0,
            **{col: [] for col in ("encode_ms", "decode_ms", "infer_ms", "post_ms", "comms_ms", "transmission_ms", "total_ms")},
        },
    )
    assert collector.build_stats_rows() == []


def test_clear_resets_results():
    collector = BenchmarkCollector()
    collector.record("zmq", {"encode_ms": 1.0, "total_ms": 10.0})
    assert collector.bench_results
    collector.clear()
    assert collector.bench_results == {}


def test_snapshot_for_heartbeat_returns_active_backend_and_bench_rows():
    """Heartbeat carries the active backend label plus the full per-backend breakdown,
    so the central UI never loses data when the operator switches transports."""
    collector = BenchmarkCollector()
    for total in (10.0, 20.0, 30.0):
        collector.record(
            "grpc",
            {"encode_ms": 1.0, "decode_ms": 1.0, "infer_ms": 5.0, "post_ms": 0.5, "total_ms": total},
        )
    collector.record(
        "imagezmq",
        {"encode_ms": 0.0, "decode_ms": 0.0, "infer_ms": 6.0, "post_ms": 0.5, "total_ms": 25.0},
    )

    out = collector.snapshot_for_heartbeat("imagezmq")
    assert out["backend"] == "imagezmq"
    rows_by_backend = {r["Backend"]: r for r in out["bench_rows"]}
    # Both backends present even though only imagezmq is currently active.
    assert set(rows_by_backend) == {"grpc", "imagezmq"}
    assert rows_by_backend["grpc"]["Frames"] == 3
    assert rows_by_backend["imagezmq"]["Frames"] == 1


def test_snapshot_for_heartbeat_with_no_data_yields_empty_rows():
    collector = BenchmarkCollector()
    out = collector.snapshot_for_heartbeat(None)
    assert out == {"backend": None, "bench_rows": []}
