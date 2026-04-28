from inference_streaming_benchmark.multi_run import SweepConfig, build_plan, run_sweep


class _Response:
    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


class _Session:
    def __init__(self):
        self.posts = []
        self.gets = []

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json, timeout))
        if url.endswith("/batching"):
            return _Response(
                {
                    "enabled": json["enabled"],
                    "max_batch_size": json["max_batch_size"],
                    "max_wait_ms": json["max_wait_ms"],
                }
            )
        if url.endswith("/clients/control-all"):
            return _Response({"results": {"client-1": "ok"}})
        if url.endswith("/clients/clear-all"):
            return _Response({"results": {"client-1": "ok"}})
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url, timeout=None):
        self.gets.append((url, timeout))
        if url.endswith("/clients"):
            return _Response(
                {
                    "active_transport": "grpc",
                    "clients": [
                        {
                            "name": "client-1",
                            "stats": {
                                "backend": "grpc",
                                "bench_rows": [{"Backend": "grpc", "Frames": 10}],
                            },
                        }
                    ],
                }
            )
        raise AssertionError(f"unexpected GET {url}")


def test_build_plan_includes_batch_off_once_per_transport():
    plan = build_plan(["grpc", "websocket"], ["off", "on"], [2, 4], [5.0, 10.0])

    assert plan[0] == SweepConfig("grpc", False, 1, 0.0)
    assert SweepConfig("grpc", True, 2, 5.0) in plan
    assert SweepConfig("grpc", True, 4, 10.0) in plan
    assert plan[5] == SweepConfig("websocket", False, 1, 0.0)
    assert len(plan) == 10


def test_run_sweep_applies_config_clears_and_collects_clients():
    session = _Session()
    sleeps = []
    completed = []

    result = run_sweep(
        [SweepConfig("grpc", True, 4, 10.0)],
        control_base="http://control",
        duration_s=3.0,
        warmup_s=0.5,
        session=session,
        sleep=sleeps.append,
        on_run_complete=completed.append,
    )

    assert sleeps == [0.5, 3.0]
    assert session.posts[0][0] == "http://control/batching"
    assert session.posts[0][1] == {"enabled": True, "max_batch_size": 4, "max_wait_ms": 10.0}
    assert session.posts[1][0] == "http://control/clients/control-all"
    assert session.posts[1][1] == {"backend": "grpc", "inference": True}
    assert session.posts[2][0] == "http://control/clients/clear-all"
    assert session.gets[0][0] == "http://control/clients"
    assert result["runs"][0]["config"] == {
        "transport": "grpc",
        "batching_enabled": True,
        "max_batch_size": 4,
        "max_wait_ms": 10.0,
    }
    assert result["runs"][0]["clients"][0]["name"] == "client-1"
    assert completed == result["runs"]
