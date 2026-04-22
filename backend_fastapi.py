import uvicorn

from inference_streaming_benchmark.backend.fastapi.ai_server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
