from inference_streaming_benchmark.config import TRANSPORT_DEFAULT_PORTS

from ..registry import register
from .transport import HTTPMultipartTransport

register(HTTPMultipartTransport)
register(
    HTTPMultipartTransport,
    name="http_multipart_raw",
    display_name="HTTP raw (FastAPI, ndarray)",
    port=TRANSPORT_DEFAULT_PORTS["http_multipart_raw"],
    raw=True,
)
