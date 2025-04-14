#!/bin/bash

# ======================================
# LOAD ENVIRONMENT VARIABLES
# ======================================

# Load variables from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
else
    echo ".env file not found!"
    exit 1
fi

# ======================================
# SCRIPT
# ======================================

echo "Starting Fly proxy to database..."

# Start Fly proxy in background
fly proxy $LOCAL_PORT:5432 -a $APP_NAME &
PROXY_PID=$!

# Wait a few seconds to let proxy start
sleep 5

echo "Starting backup..."

# Export password so pg_dump uses it
export PGPASSWORD=$DB_PASSWORD

# Run pg_dump
pg_dump -h localhost -p $LOCAL_PORT -U $DB_USER -d $DB_NAME > "$BACKUP_FILE"

# Check if dump succeeded
if [ $? -eq 0 ]; then
  echo "Backup completed successfully: $BACKUP_FILE"
else
  echo "Backup FAILED"
fi

# Kill proxy
kill $PROXY_PID

echo "Proxy closed."
