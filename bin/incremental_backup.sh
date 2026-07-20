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
INCREMENTAL_DIR="${INCREMENTAL_BACKUP_DIR:-${BACKUP_DIR}/incremental}"
INCREMENTALS_TO_KEEP="${INCREMENTALS_TO_KEEP:-30}"
PYTHON_BIN="${PYTHON_BIN:-${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}}"
if [ -z "${PYTHON_BIN:-}" ] && [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || command -v python || true)}"

if [ -z "$PYTHON_BIN" ]; then
    echo "Python not found. Activate the virtualenv or set PYTHON_BIN in .env." >&2
    exit 1
fi

mkdir -p "$LOG_DIR" "$INCREMENTAL_DIR"
NOW="$(date +"%Y-%m-%d_%H-%M-%S")"
LOG_FILE="${LOG_DIR}/incremental_backup_${NOW}.log"
OUTPUT_FILE="${INCREMENTAL_DIR}/incremental_${NOW}.jsonl.gz"
STATE_FILE="${INCREMENTAL_DIR}/state.json"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "================ $(date) ================"
echo "Starting incremental backup"
echo "Output: $OUTPUT_FILE"
echo "State:  $STATE_FILE"

cd "$ROOT_DIR"
"$PYTHON_BIN" manage.py export_incremental --state "$STATE_FILE" --output "$OUTPUT_FILE"

find "$INCREMENTAL_DIR" -name 'incremental_*.jsonl.gz' -type f -printf '%T@ %p\n' \
    | sort -rn \
    | tail -n +"$((INCREMENTALS_TO_KEEP + 1))" \
    | cut -d' ' -f2- \
    | xargs -r rm --

find "$LOG_DIR" -name 'incremental_backup_*.log' -type f -printf '%T@ %p\n' \
    | sort -rn \
    | tail -n +"$((INCREMENTALS_TO_KEEP + 1))" \
    | cut -d' ' -f2- \
    | xargs -r rm --

echo "Incremental backup completed"
