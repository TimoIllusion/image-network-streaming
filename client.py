import os

# Must precede any cv2 import in this process — disables MSMF HW transforms on Windows.
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import uvicorn  # noqa: E402

from inference_streaming_benchmark.client.app import app  # noqa: E402
from inference_streaming_benchmark.config import UI_PORT  # noqa: E402

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=UI_PORT)
