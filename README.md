# inference-streaming-benchmark

Comparison implementation for image transmission and inference response using an AI inference server with multiple communication systems: HTTP multipart (FastAPI), ZeroMQ (ZMQ), ImageZMQ, grpc.

Results for 1920×1080 images on a MacBook Pro M2 Pro (YOLOv8n, MPS inference):

| BACKEND             | ENCODE (MS) | DECODE (MS) | INFER (MS) | POST (MS) | COMMS (MS) | TOTAL (MS) | TOTAL W/O INFER (MS) |
|---------------------|-------------|-------------|------------|-----------|------------|------------|----------------------|
| imagezmq            | 0.0         | 0.0         | 25.5       | 1.6       | 3.0        | 30.3       | 4.6                  |
| zmq_raw             | 0.3         | 0.0         | 25.2       | 1.1       | 3.5        | 30.1       | 5.0                  |
| http_multipart_raw  | 0.2         | 0.0         | 24.3       | 1.7       | 7.8        | 34.3       | 9.9                  |
| grpc                | 0.2         | 0.4         | 24.3       | 1.2       | 9.2        | 35.5       | 11.1                 |
| zmq                 | 5.7         | 9.0         | 24.3       | 1.2       | 0.8        | 41.2       | 16.8                 |
| http_multipart      | 5.8         | 8.8         | 23.1       | 1.0       | 3.5        | 42.3       | 19.3                 |
| websocket           | 5.8         | 8.8         | 22.9       | 0.9       | 16.7       | 55.1       | 32.3                 |
| websocket_raw       | 0.2         | 0.0         | 23.5       | 1.0       | 166.0      | 191.0      | 167.3                |


## Setup

Requirements: Python 3.10+. [uv](https://docs.astral.sh/uv/) is recommended — it reads `pyproject.toml` and `uv.lock` for a reproducible install.

```bash
git clone https://github.com/TimoIllusion/inference-streaming-benchmark.git
cd inference-streaming-benchmark

# Recommended (uv, locked via uv.lock):
uv sync --all-extras

# Or classic pip:
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,test]"
```

>Note: Install torch with the respective hardware acceleration, e.g. CUDA to increase inference speed.

## Run

One AI server process hosts every transport. The frontend picks which protocol is active; the server hot-swaps listeners on demand. Exactly one transport is active at any time.

```bash
python serve.py        # control plane on :9000, http_multipart active by default
python frontend.py     # http://127.0.0.1:8501 (webcam UI, backend dropdown)
```

Options:
- `python serve.py --default zmq` — start with `zmq` active
- `python serve.py --default none` — idle until the frontend picks a transport

| Transport      | Port  | Description                       |
| -------------- | ----- | --------------------------------- |
| http_multipart | 8008  | FastAPI + multipart/form-data     |
| zmq            | 5555  | ZeroMQ REQ/REP + JPEG             |
| imagezmq       | 5556  | ImageZMQ + raw ndarray            |
| grpc           | 50051 | gRPC unary + raw ndarray bytes    |

>Note: transfer speed for images can be significantly boosted by resizing them before sending. This will usually not cause issues with the ai model, since most models need images of low input sizes like 224x224.

### Adding a new transport

1. Create `inference_streaming_benchmark/transports/<name>/transport.py` with a `class XxxTransport(Transport)` implementing `start` / `stop` / `connect` / `send` / `disconnect`.
2. In `inference_streaming_benchmark/transports/<name>/__init__.py`, add `register(XxxTransport)`.
3. Add `from . import <name>` to `inference_streaming_benchmark/transports/__init__.py`.

That's it — the frontend dropdown and server control plane pick it up automatically.

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
# Runs `python serve.py` (http_multipart active by default).
# Expose :9000 (control plane) plus whichever transport port you want to reach.
docker run -it --name aiserver1 --rm --shm-size=8g --gpus=all \
  -p 9000:9000 -p 8008:8008 \
  inference-streaming-benchmark:latest

# To start with a different default transport, pass through the flag and expose its port:
docker run -it --rm --shm-size=8g --gpus=all \
  -p 9000:9000 -p 5556:5556 \
  inference-streaming-benchmark:latest python serve.py --default imagezmq
```

## Versioning

Versions are derived from git tags via [setuptools-scm](https://github.com/pypa/setuptools_scm). Every merge to `main` triggers `.github/workflows/auto-tag.yml`, which creates the next tag (e.g. `v0.1.5` → `v0.1.6`).

The default bump is **patch**. Override by including one of these keywords in a commit message of the merged range:

| Keyword  | Effect                        |
| -------- | ----------------------------- |
| `#minor` | minor bump (`v0.1.5` → `v0.2.0`) |
| `#major` | major bump (`v0.1.5` → `v1.0.0`) |
| `#none`  | skip tagging for this merge   |

For squash-merged PRs, putting the keyword in the PR title or description works, since GitHub includes both in the resulting commit message. For merge commits, put it in a branch commit or edit the merge commit message.

## Common Issues

- camera not working and throwing errors -> close all open instances of streamlit in browser except one and reload it

## TODO

- [x] Dockerize ai server
- [x] Rename "fastapi" backend to a more descriptive label (e.g. `pure-http-multipart`) that reflects the protocol rather than the framework
- [x] Replace Streamlit frontend with Flask or a comparable lightweight framework for better control and lower overhead
- [x] Improve benchmark statistics: add a dedicated "transmission time" column that excludes inference and preprocessing (encode + decode) so pure transport overhead is isolated
- [ ] Extract a shared `FastAPITransport` base for the duplicated uvicorn lifecycle and `_infer` closure in `http_multipart` and `websocket`
- [ ] Centralize raw payload codec and `FRAME_SHAPE` in one module (currently redeclared in 4 transport files)
- [ ] Drop the `*_Raw` subclass pattern in favor of codec injection at registration time
- [ ] Split `frontend.py` into a `frontend/` package (camera, mjpeg, state, app)
- [ ] Decompose `FrontendState` into camera, transport-session, and benchmark-collector responsibilities
- [ ] Consolidate ports/hosts into a single env-driven config module
- [ ] Fix table layout in frontend
- [ ] Add export button to export table in ui as markdown
- [ ] Improve on the fly protocol switch and make it seamless

## AI Assistance

Development of this project was supported by AI, which provided code suggestions and troubleshooting help.

