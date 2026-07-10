#!/usr/bin/env bash
# uptime_monitor.sh — poll prices.fly.dev every minute, log status/timing, and
# push an ntfy.sh alert on DOWN/UP state transitions (not every failed check,
# to avoid spam). Pure observation + alerting only — no restart action, so it
# doesn't interact with watchdog.sh's hourly restart behavior.
# Scheduled via cron: * * * * *

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LOG="${SCRIPT_DIR}/../logs/uptime_monitor.log"
STATE_FILE="${SCRIPT_DIR}/../logs/.uptime_state"
URL="https://prices.fly.dev/"
CURL_TIMEOUT=15
NTFY_TOPIC="agile-predict-outage-7eef5967"

result=$(curl -s -o /dev/null -w '%{http_code} %{time_total}' --max-time "$CURL_TIMEOUT" "$URL" 2>/dev/null || echo '000 0')
code=$(echo "$result" | cut -d' ' -f1)
now="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

printf '%s %s\n' "$now" "$result" >> "$LOG"

if [[ "$code" =~ ^[23] ]]; then
    current_state="up"
else
    current_state="down"
fi

prev_state="up"
[ -f "$STATE_FILE" ] && prev_state="$(cat "$STATE_FILE")"

if [ "$current_state" != "$prev_state" ]; then
    if [ "$current_state" = "down" ]; then
        curl -s -H 'Title: prices.fly.dev is DOWN' -H 'Priority: urgent' -H 'Tags: rotating_light'             -d "HTTP ${code} at ${now}" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
    else
        curl -s -H 'Title: prices.fly.dev is back UP' -H 'Tags: white_check_mark'             -d "HTTP ${code} at ${now}" "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null 2>&1 || true
    fi
    echo "$current_state" > "$STATE_FILE"
fi
