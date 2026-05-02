# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Commands

**Install (development):**
```bash
uv sync --all-extras          # recommended, uses uv.lock
# or:
pip install -e ".[dev,test]"  # pip fallback
```

**Test:**
```bash
pytest tests                        # all tests
pytest tests/test_http_multipart.py # single file
pytest -q                           # quiet (used in CI)
```

**Lint & format:**
```bash
ruff check .          # lint
ruff format .         # format
ruff format --check . # CI-style format check
```

**Run the server + a client:**
```bash
python serve.py        # control plane + central UI on :9000, http_multipart active by default
python client.py       # http://127.0.0.1:8501 (per-device UI)
```

Multi-device: set `INFSB_CONTROL_HOST=<server-ip>` (and optionally `INFSB_CLIENT_NAME=<friendly>`) on each client machine. Clients auto-register on startup; central UI lives at `http://<server-ip>:9000/`.

Options: `python serve.py --default zmq` (start with zmq active), `python serve.py --default none` (idle until a client picks).

**Docker:**
```bash
docker build -f ./docker/Dockerfile -t inference-streaming-benchmark:latest .
# CMD = `python serve.py` (http_multipart active by default). Expose :9000 + transport ports you need.
docker run -it --rm --shm-size=8g --gpus=all -p 9000:9000 -p 8008:8008 inference-streaming-benchmark:latest
```

## Architecture

One AI server process hosts every transport; **exactly one transport is active at a time**. A connected client picks which protocol is live and the server hot-swaps listeners on demand.

**Disaggregated by design:** the server can run on one machine and N clients on others over the LAN. Every transport binds `0.0.0.0`. Set `INFSB_CONTROL_HOST` on each client to point at the server.

**Transport abstraction:** All transports implement `Transport` (`inference_streaming_benchmark/transports/base.py`) with five methods: `start(port, handler)` / `stop()` for the server role, and `connect(host, port)` / `send(frame)` / `disconnect()` for the client role. One class per protocol — same class used both sides.

**Registry:** `inference_streaming_benchmark/transports/registry.py`. Each transport package's `__init__.py` is one line: `register(XxxTransport)`. Importing `inference_streaming_benchmark.transports` populates the registry for the whole process.

**Server (`inference_streaming_benchmark/server.py`):** The `Server` class owns a single `InferenceEngine` (YOLO, loaded lazily) and the currently-active transport. `Server.switch(name)` tears down the active listener and starts the requested one. The FastAPI control plane on port 9000 exposes:
- `GET /health`, `GET /transports`, `POST /switch` — original control plane
- `POST /register`, `POST /heartbeat`, `GET /clients`, `POST /clients/{name}/control` — client registry
- `GET /` — central operator UI page (`server_static/`)
- `POST /switch` accepts `cascade: true` to push the new transport to every registered client.

**Client registry (`inference_streaming_benchmark/client_registry.py`):** in-memory thread-safe registry of connected clients. Stale entries (no heartbeat for >10s) age out lazily on `list_active`.

**Entry point (`serve.py`):** Kicks off `Server` + control plane; `--default` flag picks the initial transport (or `none` for idle).

**Client (`client.py` + `inference_streaming_benchmark/client/`):** FastAPI app serving a per-device static page + MJPEG video feed. On control-plane requests from the browser it forwards `POST /switch` to the server, then connects a local `Transport` client using `registry.get(name)()`. The frame loop calls `client.send(frame)` and draws detections + FPS overlays. On startup, `Registrar` (`client/registration.py`) auto-registers with the server and sends a heartbeat every second carrying current stats. The mock-camera mode is a runtime toggle (per-device UI checkbox or remote control via the central UI).

**Engine (`inference_streaming_benchmark/engine.py`):** Holds `InferenceEngine.infer(ndarray) -> (detections, timings)` and the shared `decode_jpeg_bytes` helper.

**gRPC generated files:** `*_pb2.py`, `*_pb2.pyi`, and `*_pb2_grpc.py` under `inference_streaming_benchmark/transports/grpc/` are auto-generated from the proto definition — do not edit manually. Regenerate with `uv run python -m grpc_tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. inference_streaming_benchmark/transports/grpc/ai_server.proto`.

**Adding a new transport:** create `transports/<name>/transport.py` with a `Transport` subclass, add `register(...)` in its `__init__.py`, add `from . import <name>` to `transports/__init__.py`. No edits to `server.py`, `client.py`, or any other transport.

**Per-transport multi-client behavior:** HTTP multipart / gRPC / WebSocket fan in concurrently and serialize at the shared YOLO instance. ZMQ REQ/REP and ImageZMQ are 1:1 by design — under N-client load they queue strictly. This is intentional ("Option 1") so the benchmark surfaces the real protocol behavior; do not paper it over with a shared queue.

**CI constraints:** Tests run without GPU or model weights. The YOLO model is loaded lazily (on first `engine.infer` call), so imports stay lightweight. Tests inject a fake handler to avoid weights + inference.

**Logging:** Loguru, writes to `logs/<timestamp>.log` (UTC), configured in `inference_streaming_benchmark/logging.py`.

**Ruff config:** line length 128, rules E/F/I/B/UP; excludes `.venv/`, `Ultralytics/`, and generated `*_pb2*.py` / `.pyi` files. Per-file ignores: E402 in tests, B008 in `transports/http_multipart/transport.py` (FastAPI `File(...)` default).

## Implementation Policy

- Only implement backwards compatibility after explicit confirmation from the user (ask first).

## Commit Tags (Versioning)

Every push to `main` triggers `.github/workflows/auto-tag.yml`, which bumps the version tag based on a marker in the commit message. The package version is derived from git tags via `setuptools_scm` — tags are the single source of truth.

**Every commit MUST include exactly one tag.** The agent selects it based on what changed:

- `#none` — no runtime behavior change. Docs, README, comments, AGENTS.md/CLAUDE.md, beads metadata, formatting-only diffs, CI tweaks that don't affect the package, screenshot updates.
- `#patch` — bug fix or small refactor that changes runtime behavior without adding capability.
- `#minor` — new feature or user-visible capability (new transport, new CLI flag, new metric, new UI surface).
- `#major` — breaking change. Rare; confirm with the user before using.

**Rules:**
- Never commit without a tag. The default would be `#patch`, which produces noisy release tags for trivial changes.
- When unsure between two tiers, pick the lower one (`#none` over `#patch`, `#patch` over `#minor`).
- Place the tag at the end of the subject line: `Add foo bar #minor`.
- Tag the actual content: a one-line README fix is `#none` even if it ships alongside larger work in a different PR.

## Branch & PR Conventions

**Workflow:** GitHub Flow. Branch off `main`, open a PR, squash-merge back. `main` stays deployable at all times.

**Branch names:** `<type>/<short-kebab-description>`, lowercase, ≤50 chars. Types: `feat/`, `fix/`, `refactor/`, `docs/`, `chore/`, `test/`, `ci/`, `hotfix/`. Example: `feat/grpc-streaming`.

**Commit subject:** plain imperative, sentence case, no trailing period, ≤72 chars including the `#tag` (see [Commit Tags](#commit-tags-versioning)). Example: `Add websocket transport #minor`.

**Commit body** (optional): blank line after subject, wrap at 72 chars, explain *why* not *what*. One commit = one logical change.

**PR title:** same rules as commit subject, including the `#tag` — squash-merge uses the PR title as the commit on `main`, and `auto-tag.yml` reads the tag from it.

**PR description:**
```
**Why:** <motivation; link bd-XXX if applicable>
**What:** <bullet list of changes>
**Test:** <commands run, e.g. "ruff + pytest pass">
```

**Merge strategy:** squash and merge only. Reference beads issues as `bd-XXX` (not `#XXX`, which GitHub auto-links to PR/issue numbers).

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** - Always run `uv run ruff check .` before every commit and push. If code changed, also run the relevant tests/builds.
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
