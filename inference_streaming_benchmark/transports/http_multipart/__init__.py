from ..registry import register
from .transport import HTTPMultipartRawTransport, HTTPMultipartTransport

register(HTTPMultipartTransport)
register(HTTPMultipartRawTransport)
