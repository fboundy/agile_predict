#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
OLD_ROOT="/home/boundys/data4/django/agile_predict"

ENV_FILE="${ROOT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    # Strip Windows CRLF endings and tolerate spaces around "=".
    # shellcheck disable=SC1090
    . <(sed -e 's/\r$//' -e 's/^\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=[[:space:]]*/\1=/' "$ENV_FILE")
    set +a
fi

BACKUP_DIR="${BACKUP_DIR:-${ROOT_DIR}/.local}"
BACKUP_DIR="${BACKUP_DIR/#$OLD_ROOT/$ROOT_DIR}"
BACKUP_FILE="${1:-${BACKUP_DIR}/backup.sql}"
SQLITE_DB="${2:-${ROOT_DIR}/db.sqlite3}"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
if [ -z "$PYTHON_BIN" ] && [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"

if [ -z "$PYTHON_BIN" ]; then
    echo "Python not found. Activate the virtualenv or set PYTHON_BIN in .env."
    exit 1
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

if [ -f "$SQLITE_DB" ]; then
    SNAPSHOT="${SQLITE_DB}.$(date +"%Y-%m-%d_%H-%M-%S").bak"
    cp "$SQLITE_DB" "$SNAPSHOT"
    echo "Saved existing SQLite database to: $SNAPSHOT"
fi

echo "Restoring PostgreSQL dump to SQLite"
echo "Backup: $BACKUP_FILE"
echo "SQLite: $SQLITE_DB"

cd "$ROOT_DIR"
export DATABASE_URL="sqlite:///${SQLITE_DB}"
"$PYTHON_BIN" bin/restore_sqlite.py "$BACKUP_FILE" "$SQLITE_DB"
