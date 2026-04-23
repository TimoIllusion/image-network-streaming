import uvicorn

from inference_streaming_benchmark.backend.fastapi.ai_server import app
from inference_streaming_benchmark.backend.sidecar import start_sidecar_in_thread

if __name__ == "__main__":
    start_sidecar_in_thread(transport="fastapi", data_port=8008, sidecar_port=9001)
    uvicorn.run(app, host="0.0.0.0", port=8008)
