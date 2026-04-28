# inference-streaming-benchmark

Comparison implementation for image transmission and inference response using an AI inference server with multiple communication systems: HTTP multipart (FastAPI), ZeroMQ (ZMQ), ImageZMQ, grpc.

Results for 1920×1080 images on a MacBook Pro M2 Pro (YOLOv8n, MPS inference).

The columns are ordered so the math reads left-to-right:

```
enc + dec + comms  =  transport     (network + codec overhead)
                +  infer + post  =  total          (server-side processing)
```

`post` here is the server-side JSON serialization of detection results for the response, **not** YOLO post-processing (which is already counted inside `infer`).

| Backend            | Frames | Duration (s) | FPS  | enc (ms) | dec (ms) | comms (ms) | **transport (ms)** | infer (ms) | post (ms) | **total (ms)** |
| ------------------ | ------ | ------------ | ---- | -------- | -------- | ---------- | ------------------ | ---------- | --------- | -------------- |
| imagezmq           | 149    | 4.5          | 33.0 | 0.0      | 0.0      | 3.1        | 3.1                | 26.0       | 1.3       | 30.6           |
| zmq_raw            | 81     | 2.5          | 31.8 | 0.2      | 0.0      | 3.6        | 4.1                | 26.1       | 1.2       | 31.4           |
| grpc               | 93     | 6.1          | 15.4 | 0.2      | 0.2      | 6.6        | 7.2                | 22.4       | 0.9       | 30.4           |
| websocket_raw      | 271    | 9.3          | 29.1 | 0.3      | 0.0      | 7.2        | 7.6                | 25.0       | 1.3       | 34.2           |
| http_multipart_raw | 135    | 4.6          | 29.3 | 0.2      | 0.0      | 7.6        | 7.9                | 24.0       | 1.2       | 33.2           |
| http_multipart     | 115    | 5.1          | 22.4 | 5.6      | 8.6      | 3.8        | 18.4               | 23.9       | 1.1       | 43.4           |
| zmq                | 117    | 5.2          | 22.5 | 6.6      | 11.0     | 0.9        | 18.6               | 24.3       | 1.4       | 44.4           |
| websocket          | 78     | 3.5          | 22.6 | 6.6      | 11.0     | 1.0        | 18.8               | 23.9       | 1.3       | 44.0           |


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

One AI server process hosts every transport. A connected client picks which protocol is active; the server hot-swaps listeners on demand. Exactly one transport is active at any time.

```bash
python server.py        # control plane + central UI on :9000, http_multipart active by default
python client.py        # http://127.0.0.1:8501 (webcam UI, backend dropdown, mock-camera toggle)
```

Open `http://127.0.0.1:9000/` for the **central operator panel** — registers connected clients, switches transport for everyone with one click, toggles per-client mock camera and inference.

**Multiple clients on one machine**: pass `--port` (or `--port 0` for auto-pick). The default client name auto-suffixes the port (`<hostname>-<port>`) so they don't collide in the registry:

```bash
python client.py                          # 8501, name = <hostname>-8501
python client.py --port 8502              # 8502, name = <hostname>-8502
MOCK_CAMERA=1 python client.py --port 0   # any free port, mock frames
```

Options:
- `python server.py --default zmq` — start with `zmq` active
- `python server.py --default none` — idle until a client picks a transport

| Transport      | Port  | Description                       |
| -------------- | ----- | --------------------------------- |
| http_multipart | 8008  | FastAPI + multipart/form-data     |
| zmq            | 5555  | ZeroMQ REQ/REP + JPEG             |
| imagezmq       | 5556  | ImageZMQ + raw ndarray            |
| grpc           | 50051 | gRPC unary + raw ndarray bytes    |

>Note: transfer speed for images can be significantly boosted by resizing them before sending. This will usually not cause issues with the ai model, since most models need images of low input sizes like 224x224.

## Multi-device deployment

The server (workstation/GPU box) and clients (RPi5s, laptops, anything with Python) can live on different machines on the same LAN.

**Server (workstation):**
```bash
python server.py
# central UI: http://<server-ip>:9000/
```

**Each client device (one per machine):**
```bash
INFSB_CONTROL_HOST=<server-ip> INFSB_CLIENT_NAME=rpi-edge-1 python client.py
# per-device UI: http://<client-ip>:8501/
```

Clients auto-register with the server on startup and send a heartbeat every second; the central UI shows them in a live table with FPS, latency, and per-client toggles.

**No webcam attached?** Set `MOCK_CAMERA=1` to start with synthetic frames (a static DALL-E image, looped at 30fps), or toggle the **Mock camera** switch in the per-device UI / central UI at runtime — no restart needed.

**Per-transport multi-client behavior** (Option 1: each transport keeps its natural semantics):
- HTTP multipart, gRPC, WebSocket — fan in concurrently at the network layer; serialize at inference (one shared YOLO instance).
- ZMQ REQ/REP, ImageZMQ — strictly 1:1 by design. With N clients connected, the server processes one client at a time and the others queue. The central UI surfaces this clearly: one client at full FPS, others starved.

### Dynamic batching

The server has a built-in batcher in front of the inference engine. When **enabled**, concurrent `infer()` calls are coalesced into a single `model([img1, img2, ...])` call so the GPU gets fed `N` frames per invocation instead of `1`. Aggregate throughput rises with the number of clients sending at once.

Toggle from the **central UI** (top control card): `Dynamic batching` switch + `max size` + `max wait (ms)` + Apply. Or from the API: `POST /batching {enabled, max_batch_size, max_wait_ms}`. Disabled by default — pass-through to `engine.infer` with zero overhead, so existing single-stream numbers stay valid.

Two new columns appear in every stats table:
- **`wait (ms)`** — how long each frame sat in the batcher's queue before the model call started. Zero when batching is off.
- **`batch`** — median batch size across the frames recorded for that backend. `1` when batching is off or only one client is in flight.

The `transport (ms)` column is `total − infer − post − wait`, so it stays "network + codec only" even with batching on.

**Caveat for ZMQ/ImageZMQ:** these transports submit one frame at a time (REQ/REP semantics), so the batcher never coalesces — it just adds the `max_wait_ms` timeout penalty. The `batch` column staying at `1` and `wait (ms)` matching `max_wait_ms` makes this visible. Leave batching off for those transports unless you're explicitly studying the trade-off.

| Env var               | Default              | Purpose                                      |
| --------------------- | -------------------- | -------------------------------------------- |
| `INFSB_CONTROL_HOST`  | `localhost`          | Server hostname/IP the client should reach   |
| `INFSB_CONTROL_PORT`  | `9000`               | Server control plane + central UI port       |
| `INFSB_UI_PORT`       | `8501` (auto-fallback) | Per-device client UI port. CLI `--port` overrides. |
| `INFSB_CLIENT_NAME`   | `<hostname>-<port>`    | Friendly name shown in the central UI. CLI `--name` overrides. |
| `MOCK_CAMERA`         | unset                | `1` → start with synthetic frames           |
| `INFSB_BATCH_ENABLED` | `0`                  | `1` → start the server with dynamic batching on |
| `INFSB_BATCH_SIZE`    | `8`                  | Max frames per batched model call            |
| `INFSB_BATCH_WAIT_MS` | `10`                 | Max time the batcher waits to fill a batch  |

### Adding a new transport

1. Create `inference_streaming_benchmark/transports/<name>/transport.py` with a `class XxxTransport(Transport)` implementing `start` / `stop` / `connect` / `send` / `disconnect`.
2. In `inference_streaming_benchmark/transports/<name>/__init__.py`, add `register(XxxTransport)`.
3. Add `from . import <name>` to `inference_streaming_benchmark/transports/__init__.py`.

That's it — the client backend dropdown and server control plane pick it up automatically.

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
# Runs `python server.py` (http_multipart active by default).
# Expose :9000 (control plane) plus whichever transport port you want to reach.
docker run -it --name aiserver1 --rm --shm-size=8g --gpus=all \
  -p 9000:9000 -p 8008:8008 \
  inference-streaming-benchmark:latest

# To start with a different default transport, pass through the flag and expose its port:
docker run -it --rm --shm-size=8g --gpus=all \
  -p 9000:9000 -p 5556:5556 \
  inference-streaming-benchmark:latest python server.py --default imagezmq
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
- [x] Split `frontend.py` into a package (camera, mjpeg, state, app) — now `inference_streaming_benchmark/client/`
- [x] Multi-device deployment + central operator panel
- [x] Extract a shared `FastAPITransport` base for the duplicated uvicorn lifecycle in `http_multipart` and `websocket` (`transports/_fastapi_base.py`)
- [x] Centralize raw payload codec and `FRAME_SHAPE` in one module (`transports/codec.py`)
- [x] Drop the `*_Raw` subclass pattern in favor of codec injection at registration time
- [ ] Consolidate transport `default_port` constants into the env-driven config module (still per-class today)
- [x] **Dynamic batching on the inference engine** — server-side batcher coalesces concurrent requests into a single `model([...])` call. Toggle + tune from the central UI; new `wait (ms)` + `batch` columns surface the effect in every stats table.
- [ ] Improve logging for batching and inference in general

## AI Assistance

Development of this project was supported by AI, which provided code suggestions and troubleshooting help.

