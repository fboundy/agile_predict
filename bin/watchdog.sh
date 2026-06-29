#!/usr/bin/env bash
# watchdog.sh — poll prices.fly.dev; restart fly machines if unresponsive
# Scheduled via cron: 45 * * * *

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LOG="${SCRIPT_DIR}/../logs/watchdog.log"
APP="prices"
URL="https://prices.fly.dev/"
MAX_ATTEMPTS=3
CURL_TIMEOUT=25

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }

http_code() {
    curl -s -o /dev/null -w "%{http_code}" --max-time "$CURL_TIMEOUT" "$URL" 2>/dev/null || echo "000"
}

code="000"
for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    code=$(http_code)
    if [[ "$code" =~ ^[23] ]]; then
        log "OK http=$code"
        exit 0
    fi
    log "WARN attempt=$attempt http=$code"
    [ "$attempt" -lt "$MAX_ATTEMPTS" ] && sleep 15
done

log "DOWN after $MAX_ATTEMPTS attempts (last http=$code) — restarting $APP"
if fly apps restart "$APP" >> "$LOG" 2>&1; then
    log "Restart issued"
else
    log "ERROR fly apps restart failed (exit $?)"
fi
