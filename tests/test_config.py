import importlib

from inference_streaming_benchmark import config


def test_transport_default_ports_preserve_existing_values():
    assert config.TRANSPORT_DEFAULT_PORTS == {
        "http_multipart": 8008,
        "http_multipart_raw": 8010,
        "zmq": 5555,
        "zmq_raw": 5557,
        "websocket": 8009,
        "websocket_raw": 8011,
        "imagezmq": 5556,
        "grpc": 50051,
    }


def test_transport_default_ports_can_be_overridden_from_env(monkeypatch):
    monkeypatch.setenv("INFSB_TRANSPORT_PORT_HTTP_MULTIPART", "18008")
    monkeypatch.setenv("INFSB_TRANSPORT_PORT_ZMQ_RAW", "15557")

    reloaded = importlib.reload(config)
    try:
        assert reloaded.TRANSPORT_DEFAULT_PORTS["http_multipart"] == 18008
        assert reloaded.TRANSPORT_DEFAULT_PORTS["zmq_raw"] == 15557
        assert reloaded.TRANSPORT_DEFAULT_PORTS["grpc"] == 50051
    finally:
        importlib.reload(config)
