#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"

URL="http://localhost:${PORT}/?track=monaco&cars=30&speed=1&mutation=0.08&timeout=1&timeoutFrames=3500&fresh=1"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

python3 -m http.server "${PORT}" >/dev/null 2>&1 &
SERVER_PID="$!"

# Give the server a moment to bind.
sleep 0.2

if command -v open >/dev/null 2>&1; then
  open "${URL}"
else
  echo "${URL}"
fi

wait "${SERVER_PID}"

