#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

HOST="${DJANGO_RUNSERVER_HOST:-0.0.0.0}"
PORT="${DJANGO_RUNSERVER_PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/runserver.log"

mkdir -p "$LOG_DIR"

if pgrep -af "manage.py runserver ${HOST}:${PORT}" >/dev/null; then
    echo "Django runserver is already running on ${HOST}:${PORT}"
    exit 0
fi

cd "$ROOT_DIR"
nohup "$PYTHON_BIN" manage.py runserver "${HOST}:${PORT}" >> "$LOG_FILE" 2>&1 < /dev/null &
echo "Started Django runserver on ${HOST}:${PORT}"
