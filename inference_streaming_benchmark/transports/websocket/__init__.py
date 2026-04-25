from ..registry import register
from .transport import WebSocketTransport

register(WebSocketTransport)
register(
    WebSocketTransport,
    name="websocket_raw",
    display_name="WebSocket raw (sync, ndarray)",
    port=8011,
    raw=True,
)
