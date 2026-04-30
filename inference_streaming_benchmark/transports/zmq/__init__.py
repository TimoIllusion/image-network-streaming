from inference_streaming_benchmark.config import TRANSPORT_DEFAULT_PORTS

from ..registry import register
from .transport import ZMQTransport

register(ZMQTransport)
register(
    ZMQTransport,
    name="zmq_raw",
    display_name="ZeroMQ REQ/REP (raw ndarray)",
    port=TRANSPORT_DEFAULT_PORTS["zmq_raw"],
    raw=True,
)
