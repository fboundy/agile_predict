#!/usr/bin/env bash
# watchdog.sh — poll prices.fly.dev; restart fly machines if unresponsive
# Scheduled via cron: */10 * * * *  (outages now cascade and clear in minutes,
# so an hourly check left the site down far longer than the incident itself)

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG="${SCRIPT_DIR}/../logs/watchdog.log"
APP="prices"
URL="https://prices.fly.dev/"
MAX_ATTEMPTS=3
CURL_TIMEOUT=25

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }

# flyctl needs an API token: under cron there is no interactive CLI session,
# and the CLI's own session token expires — which silently disabled this
# watchdog and left a 6-hour outage unhealed on 2026-08-04. Take a long-lived
# deploy-scoped token from .env (FLY_DEPLOY_TOKEN) so restarts always work.
if [ -f "${ROOT_DIR}/.env" ]; then
    _tok="$(grep -E '^FLY_DEPLOY_TOKEN=' "${ROOT_DIR}/.env" | head -1 | cut -d= -f2- | tr -d '"'"'"'')"
    [ -n "${_tok:-}" ] && export FLY_API_TOKEN="$_tok"
fi
export PATH="$PATH:/home/agile/.fly/bin"

http_code() {
    # -4 (IPv4-only): avoids the CT router's ~5s AAAA-lookup stall on sparse queries.
    curl -4 -s -o /dev/null -w "%{http_code}" --max-time "$CURL_TIMEOUT" "$URL" 2>/dev/null || echo "000"
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
