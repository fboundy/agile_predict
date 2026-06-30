#!/bin/bash

# Timestamp for logs and backup file naming
NOW=$(date +"%Y-%m-%d_%H-%M-%S")

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

# Load environment. Strip Windows CRLF endings and tolerate spaces around "=".
ENV_FILE="${ROOT_DIR}/.env"
if [ -f "$ENV_FILE" ]; then
    set -a
    . <(sed -e 's/\r$//' -e 's/^\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*=[[:space:]]*/\1=/' "$ENV_FILE")
    set +a
else
    echo "Missing .env file"
    exit 1
fi

# Ensure LOG_DIR, BACKUP_DIR, and BACKUPS_TO_KEEP are set.
# Older deployments used this absolute root; map those paths to this checkout.
OLD_ROOT="/home/boundys/data4/django/agile_predict"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs}"
BACKUP_DIR="${BACKUP_DIR:-${ROOT_DIR}/.local}"
LOG_DIR="${LOG_DIR/#$OLD_ROOT/$ROOT_DIR}"
BACKUP_DIR="${BACKUP_DIR/#$OLD_ROOT/$ROOT_DIR}"
BACKUPS_TO_KEEP="${BACKUPS_TO_KEEP:-3}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/backup_$NOW.log"

# Redirect all output to log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "================ $(date) ================"
echo "Starting backup script..."

# Ensure PATH includes common locations for fly and pg_dump
export PATH="$HOME/.fly/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

FLY_BIN="${FLY_BIN:-$(command -v fly || true)}"
PG_DUMP_BIN="${PG_DUMP_BIN:-$(command -v pg_dump || true)}"
NC_BIN="${NC_BIN:-$(command -v nc || true)}"

if [ -z "$FLY_BIN" ]; then
    echo "fly command not found. Install flyctl or set FLY_BIN in .env."
    exit 1
fi

if [ -z "$PG_DUMP_BIN" ]; then
    echo "pg_dump command not found. Install postgresql-client or set PG_DUMP_BIN in .env."
    exit 1
fi

if [ -z "$NC_BIN" ]; then
    echo "nc command not found. Install netcat-openbsd or set NC_BIN in .env."
    exit 1
fi

echo "APP_NAME: $APP_NAME"
echo "LOCAL_PORT: $LOCAL_PORT"
echo "Starting Fly proxy to database..."

# Start Fly proxy
"$FLY_BIN" proxy "${LOCAL_PORT}:5432" -a "$APP_NAME" &
PROXY_PID=$!

# Wait for proxy to be ready
for i in {1..10}; do
    if "$NC_BIN" -z localhost "$LOCAL_PORT"; then
        echo "Proxy is ready!"
        break
    fi
    echo "Waiting for proxy to be ready..."
    sleep 1
done

# If proxy didn't start
if ! "$NC_BIN" -z localhost "$LOCAL_PORT"; then
    echo "Proxy failed to start on port $LOCAL_PORT"
    if kill -0 $PROXY_PID 2>/dev/null; then
        kill $PROXY_PID
    fi
    exit 1
fi

echo "Starting backup..."
export PGPASSWORD=$DB_PASSWORD

mkdir -p "$BACKUP_DIR"
BACKUP_FILE="$BACKUP_DIR/${DB_NAME}_backup_$NOW.sql"
LATEST_BACKUP_FILE="$BACKUP_DIR/backup.sql"

"$PG_DUMP_BIN" -h 127.0.0.1 -p "$LOCAL_PORT" -U "$DB_USER" -d "$DB_NAME" > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"
    cp "$BACKUP_FILE" "$LATEST_BACKUP_FILE"
    echo "Updated latest backup symlink: $LATEST_BACKUP_FILE"
else
    echo "Backup FAILED"
fi

# Cleanup old backups and their logs beyond $BACKUPS_TO_KEEP
BACKUP_PATTERN="$BACKUP_DIR/${DB_NAME}_backup_*.sql"
LOG_PATTERN="$LOG_DIR/backup_*.log"

BACKUP_LIST=$(ls -1t $BACKUP_PATTERN 2>/dev/null)
LOG_LIST=$(ls -1t $LOG_PATTERN 2>/dev/null)

DELETE_BACKUPS=$(echo "$BACKUP_LIST" | tail -n +$((BACKUPS_TO_KEEP + 1)))
DELETE_LOGS=$(echo "$LOG_LIST" | tail -n +$((BACKUPS_TO_KEEP + 1)))

echo "$DELETE_BACKUPS" | xargs -r rm --
echo "$DELETE_LOGS" | xargs -r rm --

echo "Old backups and logs cleaned. Keeping last $BACKUPS_TO_KEEP."

# Kill proxy if still running
if kill -0 $PROXY_PID 2>/dev/null; then
    kill $PROXY_PID
    echo "Proxy closed."
else
    echo "Proxy process already closed"
fi
