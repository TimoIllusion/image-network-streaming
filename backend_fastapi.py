import uvicorn

from image_network_streaming.backend.fastapi.ai_server import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
