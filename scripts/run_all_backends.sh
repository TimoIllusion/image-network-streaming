#!/usr/bin/env bash
# Start all four backends in the background. Ctrl+C stops them all.
set -euo pipefail

trap 'kill 0' SIGINT SIGTERM EXIT

python backend_http_multipart.py &
python backend_zmq.py &
python backend_imagezmq.py &
python backend_grpc.py &
wait
