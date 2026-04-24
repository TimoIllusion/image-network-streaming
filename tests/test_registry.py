def test_registry_populated_by_barrel_import():
    # Importing the transports package triggers every transport's registration.
    from inference_streaming_benchmark import transports  # noqa: F401
    from inference_streaming_benchmark.transports import registry

    names = set(registry.all_transports())
    assert names == {
        "http_multipart",
        "http_multipart_raw",
        "zmq",
        "zmq_raw",
        "websocket",
        "websocket_raw",
        "imagezmq",
        "grpc",
    }


def test_registry_get_returns_transport_class():
    from inference_streaming_benchmark import transports  # noqa: F401
    from inference_streaming_benchmark.transports import registry
    from inference_streaming_benchmark.transports.base import Transport

    cls = registry.get("http_multipart")
    assert issubclass(cls, Transport)
    assert cls.name == "http_multipart"
    assert cls.default_port == 8008
