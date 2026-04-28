import numpy as np
import pytest

pytest.importorskip("grpc")

from inference_streaming_benchmark.transports.grpc.ai_server_pb2 import FrameRequest
from inference_streaming_benchmark.transports.grpc.transport import _Servicer


class _Context:
    def invocation_metadata(self):
        return (("x-infsb-client", "client-grpc"), ("x-infsb-request-id", "req-grpc"))

    def set_trailing_metadata(self, metadata):
        self.trailing_metadata = metadata


def test_grpc_servicer_uses_frame_shape_and_dtype_metadata():
    frame = np.arange(4 * 5 * 1, dtype=np.uint16).reshape((4, 5, 1))
    seen = {}

    def handler(request):
        seen["shape"] = request.image.shape
        seen["dtype"] = request.image.dtype
        seen["client_name"] = request.client_name
        seen["request_id"] = request.request_id
        return [[]], {"infer_ms": 1.0, "post_ms": 0.1}

    context = _Context()
    response = _Servicer(handler).Detect(
        FrameRequest(image=frame.tobytes(), shape=list(frame.shape), dtype=str(frame.dtype)),
        context,
    )

    assert len(response.results) == 1
    assert seen == {
        "shape": frame.shape,
        "dtype": frame.dtype,
        "client_name": "client-grpc",
        "request_id": "req-grpc",
    }
