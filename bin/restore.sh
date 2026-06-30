#!/bin/bash
set -euo pipefail

ROOT_DIR="/home/boundys/data4/django/agile_predict"
BACKUP_FILE="${1:-$ROOT_DIR/.local/backup.sql}"
DBNAME="${2:-agile_predict}"
DBOWNER="${3:-${PGUSER:-$(id -un)}}"

if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

run_pg() {
    if command -v sudo >/dev/null 2>&1 && id postgres >/dev/null 2>&1 && [ "$(id -un)" != "postgres" ]; then
        sudo -u postgres "$@"
    else
        "$@"
    fi
}

echo "Restoring $BACKUP_FILE to local database $DBNAME"
echo "Dropping database '$DBNAME'..."
run_pg dropdb --if-exists "$DBNAME"

echo "Creating database '$DBNAME' owned by '$DBOWNER'..."
run_pg createdb -O "$DBOWNER" "$DBNAME"

echo "Loading backup..."
run_pg psql -d "$DBNAME" -v ON_ERROR_STOP=1 -f "$BACKUP_FILE"

echo "Restore completed!"
