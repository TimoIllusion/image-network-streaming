"""Ad-hoc smoke test for the gRPC transport.

Assumes `python serve.py --default grpc` is running on localhost:50051.
"""

import cv2

from inference_streaming_benchmark.transports.grpc.transport import GRPCTransport

img = cv2.imread(r"resources/example_dall_e.png")

client = GRPCTransport()
client.connect("localhost", 50051)
detections, timings = client.send(img)
client.disconnect()

for det in detections or []:
    print(det)
