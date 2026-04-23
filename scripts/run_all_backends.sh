#!/usr/bin/env bash
# Start all four backends in the background. Ctrl+C stops them all.
set -euo pipefail

trap 'kill 0' SIGINT SIGTERM EXIT

python backend_fastapi.py &
python backend_zmq.py &
python backend_imagezmq.py &
python backend_grpc.py &
wait
