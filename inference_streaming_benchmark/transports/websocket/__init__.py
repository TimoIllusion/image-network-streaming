from ..registry import register
from .transport import WebSocketTransport

register(WebSocketTransport)
register(
    WebSocketTransport,
    name="websocket_raw",
    display_name="WebSocket raw (FastAPI, ndarray)",
    port=8011,
    raw=True,
)
