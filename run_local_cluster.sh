#!/bin/sh

set -eu

DEFAULT_TRANSPORT="${INFSB_DEFAULT_TRANSPORT:-http_multipart}"
CLIENT_COUNT="${1:-${INFSB_CLIENT_COUNT:-3}}"
CLIENT_BASE_PORT="${INFSB_CLIENT_BASE_PORT:-8501}"
CLIENT_NAME_PREFIX="${INFSB_CLIENT_NAME_PREFIX:-local-client}"
LOG_DIR="${INFSB_CLIENT_LOG_DIR:-logs/clients}"

mkdir -p "$LOG_DIR"

SERVER_PID=""
CLIENT_PIDS=""

stop_pid() {
  pid="$1"
  pkill -TERM -P "$pid" 2>/dev/null || true
  kill "$pid" 2>/dev/null || true
}

cleanup() {
  echo
  echo "Stopping server and clients..."
  for pid in $CLIENT_PIDS; do
    stop_pid "$pid"
  done
  if [ -n "$SERVER_PID" ]; then
    stop_pid "$SERVER_PID"
  fi
  wait 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "Starting server with default transport: $DEFAULT_TRANSPORT"
PYTHONUNBUFFERED=1 uv run python server.py --default "$DEFAULT_TRANSPORT" &
SERVER_PID="$!"

sleep 1

i=1
while [ "$i" -le "$CLIENT_COUNT" ]; do
  port=$((CLIENT_BASE_PORT + i - 1))
  name="${CLIENT_NAME_PREFIX}-${i}"
  log_file="${LOG_DIR}/${name}.log"
  echo "Starting client $name on port $port -> $log_file"
  PYTHONUNBUFFERED=1 uv run python client.py --port "$port" --name "$name" >"$log_file" 2>&1 &
  CLIENT_PIDS="$CLIENT_PIDS $!"
  i=$((i + 1))
done

echo
echo "Server output is shown below. Client logs are in: $LOG_DIR"
echo "Press Ctrl-C to stop everything."
echo

wait "$SERVER_PID"
