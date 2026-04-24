# inference-streaming-benchmark

Comparison implementation for image transmission and inference response using an AI inference server with multiple communication systems: HTTP multipart (FastAPI), ZeroMQ (ZMQ), ImageZMQ, grpc.

Results for 1920×1080 images on a MacBook Pro M2 Pro (YOLOv8n, MPS inference):

| Backend        | Transmission (ms) | Inference (ms) | Total (ms) |
| -------------- | ----------------- | -------------- | ---------- |
| imagezmq       | 4.3               | 23.6           | 28.4       |
| grpc           | 11.0              | 27.1           | 37.9       |
| zmq            | 17.5              | 37.8           | 55.7       |
| http_multipart | 20.6              | 38.9           | 58.9       |


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

>Note: Each backend binds its own data port plus a small HTTP sidecar used by the frontend as a status indicator. You can run any subset (or all) of the backends at once — the frontend dropdown shows which are online.

| Transport      | Data port | Sidecar port |
| -------------- | --------- | ------------ |
| http_multipart | 8008      | 9001         |
| zmq            | 5555      | 9002         |
| imagezmq       | 5556      | 9003         |
| grpc           | 50051     | 9004         |

To start all four at once: `./scripts/run_all_backends.sh` (Ctrl+C stops them all).

**Using HTTP multipart (FastAPI + multipart/form-data) for communication**

>Note: The order of starting is important!

```bash
python backend_http_multipart.py

python frontend.py

http://127.0.0.1:8501
```

**Using ZMQ for communication**

>Note: The order of starting is important!

```bash
python backend_zmq.py

python frontend.py

http://127.0.0.1:8501
```

**Using ImageZMQ for communication**

>Note: The order of starting is important!

```bash
python backend_imagezmq.py

python frontend.py

http://127.0.0.1:8501
```

**Using GRPC for communication**

>Note: The order of starting is important!

```bash
python backend_grpc.py

python frontend.py

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

## AI Assistance

Development of this project was supported by AI, which provided code suggestions and troubleshooting help.

