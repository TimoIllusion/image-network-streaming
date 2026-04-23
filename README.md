# inference-streaming-benchmark

Comparison implementation for image transmission and inference response using an AI inference server with multiple communication systems: FastAPI, ZeroMQ (ZMQ), ImageZMQ, grpc.

Results for 1920×1080 images on a MacBook Pro M2 Pro (YOLOv8n, MPS inference):

| Backend  | Transmission (ms) | Inference (ms) | Total (ms) |
| -------- | ----------------- | -------------- | ---------- |
| ImageZMQ | 3.4               | 19.8           | 23.1       |
| ZMQ      | 8.3               | 19.6           | 32.9       |
| gRPC     | 9.3               | 19.7           | 29.3       |
| FastAPI  | 10.9              | 19.2           | 34.6       |


## Setup

Requirements: Python 3.10+ with pip and venv available. Using *uv* for easy and fast python env setup is recommended.

```bash
git clone https://github.com/TimoIllusion/inference-streaming-benchmark.git
cd inference-streaming-benchmark
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

>Note: Install torch with the respective hardware acceleration, e.g. CUDA to increase inference speed.

## Run

>Note: Each backend binds its own data port plus a small HTTP sidecar used by the frontend as a status indicator. You can run any subset (or all) of the backends at once — the frontend dropdown shows which are online.

| Transport | Data port | Sidecar port |
| --------- | --------- | ------------ |
| fastapi   | 8008      | 9001         |
| zmq       | 5555      | 9002         |
| imagezmq  | 5556      | 9003         |
| grpc      | 50051     | 9004         |

To start all four at once: `./scripts/run_all_backends.sh` (Ctrl+C stops them all).

**Using FastAPI for communication**

>Note: The order of starting is important!

```bash
python backend_fastapi.py

streamlit run frontend.py

http://127.0.0.1:8501
```

**Using ZMQ for communication**

>Note: The order of starting is important!

```bash
python backend_zmq.py

streamlit run frontend.py

http://127.0.0.1:8501
```

**Using ImageZMQ for communication**

>Note: The order of starting is important!

```bash
python backend_imagezmq.py

streamlit run frontend.py

http://127.0.0.1:8501
```

**Using GRPC for communication**

>Note: The order of starting is important!

```bash
python backend_grpc.py

streamlit run frontend.py

http://127.0.0.1:8501
```

>Note: transfer speed for images can be significantly boosted by resizing them before sending. This will usually not cause issues with the ai model, since most models need images of low input sizes like 224x224.

## Tests

```bash
pip install -e ".[test]"
```

```bash
pytest tests
```
## Docker

```bash
docker build -f ./docker/Dockerfile -t inference-streaming-benchmark:latest .
```

```bash
# runs imagezmq backend by default
docker run -it --name aiserver1 --rm --shm-size=8g --gpus=all -p 5556:5556 inference-streaming-benchmark:latest
```

## Common Issues

- camera not working and throwing errors -> close all open instances of streamlit in browser except one and reload it

## TODO

- [x] Dockerize ai server
- [ ] Rename "fastapi" backend to a more descriptive label (e.g. `pure-http-multipart`) that reflects the protocol rather than the framework
- [ ] Replace Streamlit frontend with Flask or a comparable lightweight framework for better control and lower overhead
- [ ] Improve benchmark statistics: add a dedicated "transmission time" column that excludes inference and preprocessing (encode + decode) so pure transport overhead is isolated

## AI Assistance

Development of this project was supported by AI, which provided code suggestions and troubleshooting help.

