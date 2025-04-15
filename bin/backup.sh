#!/bin/bash

# Load environment
if [ -f bin/.env ]; then
    set -a
    . bin/.env
    set +a

else
    echo "Missing .env file"
    exit 1
fi

echo "APP_NAME: $APP_NAME"
echo "LOCAL_PORT: $LOCAL_PORT"
echo "Starting Fly proxy to database..."

# Start Fly proxy
fly proxy ${LOCAL_PORT}:5432 -a $APP_NAME &
PROXY_PID=$!

# Wait for proxy to be ready
for i in {1..10}; do
    if nc -z localhost "$LOCAL_PORT"; then
        echo "Proxy is ready!"
        break
    fi
    echo "Waiting for proxy to be ready..."
    sleep 1
done

# If proxy didn't start
if ! nc -z localhost "$LOCAL_PORT"; then
    echo "Proxy failed to start on port $LOCAL_PORT"
    if kill -0 $PROXY_PID 2>/dev/null; then
        kill $PROXY_PID
    fi
    exit 1
fi


echo "Starting backup..."
export PGPASSWORD=$DB_PASSWORD

pg_dump -h localhost -p $LOCAL_PORT -U $DB_USER -d $DB_NAME > "$BACKUP_FILE"

if [ $? -eq 0 ]; then
    echo "Backup completed successfully: $BACKUP_FILE"
else
    echo "Backup FAILED"
fi

# Kill proxy if still running
if kill -0 $PROXY_PID 2>/dev/null; then
    kill $PROXY_PID
    echo "Proxy closed."
else
    echo "Proxy process already closed"
fi
