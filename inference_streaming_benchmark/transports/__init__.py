"""Transport abstraction + registry.

Importing this package triggers registration of every transport subpackage.
Callers then use :mod:`inference_streaming_benchmark.transports.registry`
to discover available transports by name.
"""

from . import (  # noqa: F401 — import triggers registration
    grpc,
    http_multipart,
    imagezmq,
    registry,  # noqa: F401 — re-exported for convenience
    websocket,
    zmq,
)
