#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/nightly_restore.log"

mkdir -p "$LOG_DIR"

restart_server() {
    "${ROOT_DIR}/bin/runserver.sh"
}

exec >> "$LOG_FILE" 2>&1

echo "================ $(date) ================"
echo "Starting nightly SQLite restore..."

trap restart_server EXIT

pkill -f "manage.py runserver" || true
sleep 2

"${ROOT_DIR}/bin/restore_sqlite.sh"

echo "Nightly SQLite restore completed."
