from inference_streaming_benchmark.config import TRANSPORT_DEFAULT_PORTS

from ..registry import register
from .transport import WebSocketTransport

register(WebSocketTransport)
register(
    WebSocketTransport,
    name="websocket_raw",
    display_name="WebSocket raw (sync, ndarray)",
    port=TRANSPORT_DEFAULT_PORTS["websocket_raw"],
    raw=True,
)
