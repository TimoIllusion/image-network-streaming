from ..registry import register
from .transport import HTTPMultipartTransport

register(HTTPMultipartTransport)
register(
    HTTPMultipartTransport,
    name="http_multipart_raw",
    display_name="HTTP raw (FastAPI, ndarray)",
    port=8010,
    raw=True,
)
