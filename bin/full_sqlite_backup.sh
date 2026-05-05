#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . <(sed -e 's/\r$//' -e 's/^\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=[[:space:]]*/\1=/' "$ENV_FILE")
    set +a
fi

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
BACKUP_DIR="${BACKUP_DIR:-${ROOT_DIR}/.local}"
OLD_ROOT="/home/boundys/data4/django/agile_predict"
LOG_DIR="${LOG_DIR/#$OLD_ROOT/$ROOT_DIR}"
BACKUP_DIR="${BACKUP_DIR/#$OLD_ROOT/$ROOT_DIR}"
FULL_BACKUP_DIR="${FULL_BACKUP_DIR:-${BACKUP_DIR}/full}"
INCREMENTAL_DIR="${INCREMENTAL_BACKUP_DIR:-${BACKUP_DIR}/incremental}"
FULL_BACKUPS_TO_KEEP="${FULL_BACKUPS_TO_KEEP:-12}"
SQLITE_DB="${SQLITE_DB:-${ROOT_DIR}/db.sqlite3}"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
if [ -z "${PYTHON_BIN:-}" ] && [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"

if [ -z "$PYTHON_BIN" ]; then
    echo "Python not found. Activate the virtualenv or set PYTHON_BIN in .env." >&2
    exit 1
fi

if [ ! -f "$SQLITE_DB" ]; then
    echo "SQLite database not found: $SQLITE_DB" >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$FULL_BACKUP_DIR" "$INCREMENTAL_DIR"
NOW="$(date +"%Y-%m-%d_%H-%M-%S")"
LOG_FILE="${LOG_DIR}/full_sqlite_backup_${NOW}.log"
BACKUP_FILE="${FULL_BACKUP_DIR}/db_${NOW}.sqlite3.gz"
LATEST_FILE="${FULL_BACKUP_DIR}/latest.sqlite3.gz"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================ $(date) ================"
echo "Starting full SQLite backup"
echo "SQLite: $SQLITE_DB"
echo "Output: $BACKUP_FILE"

SOURCE_DB="$SQLITE_DB" OUTPUT_FILE="$BACKUP_FILE" "$PYTHON_BIN" - <<'PY'
import gzip
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

source = Path(os.environ["SOURCE_DB"])
output = Path(os.environ["OUTPUT_FILE"])
output.parent.mkdir(parents=True, exist_ok=True)

with tempfile.NamedTemporaryFile(prefix="sqlite_backup_", suffix=".sqlite3", delete=False) as tmp:
    temp_path = Path(tmp.name)

try:
    source_conn = sqlite3.connect(source)
    backup_conn = sqlite3.connect(temp_path)
    with backup_conn:
        source_conn.backup(backup_conn)
    backup_conn.close()
    source_conn.close()

    with temp_path.open("rb") as src, gzip.open(output, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
finally:
    try:
        temp_path.unlink()
    except FileNotFoundError:
        pass
PY

cp "$BACKUP_FILE" "$LATEST_FILE"
echo "Updated latest full backup: $LATEST_FILE"

echo "Initializing incremental backup state after full backup"
cd "$ROOT_DIR"
"$PYTHON_BIN" manage.py export_incremental --initialize-state --state "${INCREMENTAL_DIR}/state.json"

find "$FULL_BACKUP_DIR" -name 'db_*.sqlite3.gz' -type f -printf '%T@ %p\n' \
    | sort -rn \
    | tail -n +"$((FULL_BACKUPS_TO_KEEP + 1))" \
    | cut -d' ' -f2- \
    | xargs -r rm --

find "$LOG_DIR" -name 'full_sqlite_backup_*.log' -type f -printf '%T@ %p\n' \
    | sort -rn \
    | tail -n +"$((FULL_BACKUPS_TO_KEEP + 1))" \
    | cut -d' ' -f2- \
    | xargs -r rm --

echo "Full SQLite backup completed"
