#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
LOG_FILE="${LOG_DIR}/nightly_restore.log"
SQLITE_DB="${ROOT_DIR}/db.sqlite3"
SNAPSHOT=""

mkdir -p "$LOG_DIR"

restart_server() {
    "${ROOT_DIR}/bin/runserver.sh"
}

on_exit() {
    if [ -n "$SNAPSHOT" ] && [ -f "$SNAPSHOT" ]; then
        echo "Restore failed; rolling back to: $SNAPSHOT" >&2
        cp "$SNAPSHOT" "$SQLITE_DB"
    fi
    restart_server
}

exec >> "$LOG_FILE" 2>&1

echo "================ $(date) ================"
echo "Starting nightly SQLite restore..."

# Take snapshot before stopping the server so we can roll back if restore fails.
if [ -f "$SQLITE_DB" ]; then
    SNAPSHOT="${SQLITE_DB}.$(date +"%Y-%m-%d_%H-%M-%S").nightly.bak"
    cp "$SQLITE_DB" "$SNAPSHOT"
    echo "Snapshot: $SNAPSHOT"
fi

trap on_exit EXIT

pkill -f "manage.py runserver" || true
sleep 2

"${ROOT_DIR}/bin/restore_sqlite.sh"

# Restore succeeded; clear SNAPSHOT so on_exit does not roll back.
SNAPSHOT=""
echo "Nightly SQLite restore completed."
