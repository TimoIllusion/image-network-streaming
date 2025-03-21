from image_network_streaming.backend.grpc.api import GRPCBackendInterface

import cv2

img = cv2.imread(r"resources\example_dall_e.png")

api = GRPCBackendInterface()
result = api.send_frame_to_ai_server(img)

for box in result.boxes:
    print(box)
