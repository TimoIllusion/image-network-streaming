from ..registry import register
from .transport import ZMQTransport

register(ZMQTransport)
register(
    ZMQTransport,
    name="zmq_raw",
    display_name="ZeroMQ REQ/REP (raw ndarray)",
    port=5557,
    raw=True,
)
