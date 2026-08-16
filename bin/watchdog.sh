#!/usr/bin/env bash
# watchdog.sh — poll prices.fly.dev; restart fly machines if unresponsive
# Scheduled via cron: */10 * * * *  (outages now cascade and clear in minutes,
# so an hourly check left the site down far longer than the incident itself)

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
LOG="${SCRIPT_DIR}/../logs/watchdog.log"
DIAG_LOG="${SCRIPT_DIR}/../logs/wedge_diagnostics.log"
APP="prices"
URL="https://prices.fly.dev/"
MAX_ATTEMPTS=3
CURL_TIMEOUT=25
DIAG_TIMEOUT=45

log() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG"; }
diag() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$DIAG_LOG"; }

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

# Capture the machine's internal state BEFORE restarting it. A restart destroys
# the evidence, and Fly's log retention does not reach back to the start of a
# long outage — on 2026-08-16 four hours of downtime and twenty-four restarts
# produced nothing that could distinguish blocked writes from a wedged DB. SSH
# still works when HTTP is dead, which is what makes this possible.
#
# The payload is base64'd because quoting a multi-line script through
# `fly ssh console -C` is unreliable; note also that ps/ss/free are absent from
# the image, so everything here reads /proc directly.
DIAG_PY='import os
socks = {}
for f in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(f).read().split(chr(10))[1:]
    except Exception:
        continue
    for line in lines:
        p = line.split()
        if len(p) < 5:
            continue
        try:
            if int(p[1].split(":")[1], 16) != 8000:
                continue
            tx = int(p[4].split(":")[0], 16)
        except Exception:
            continue
        e = socks.setdefault(p[3], [0, 0])
        e[0] += 1
        if tx > 0:
            e[1] += 1
print("sockets_port8000 state:[count,with_pending_tx] =", socks)
for f in ("cpu", "io", "memory"):
    try:
        print("pressure_" + f, open("/proc/pressure/" + f).read().strip().replace(chr(10), " | "))
    except Exception:
        print("pressure_" + f, "n/a")
wch = {}
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    tdir = "/proc/" + pid + "/task"
    try:
        tasks = os.listdir(tdir)
    except Exception:
        continue
    for t in tasks:
        try:
            w = open(tdir + "/" + t + "/wchan").read().strip() or "running"
        except Exception:
            continue
        wch[w] = wch.get(w, 0) + 1
print("thread_wchan", sorted(wch.items(), key=lambda kv: -kv[1])[:12])
try:
    mi = dict(l.split(":", 1) for l in open("/proc/meminfo").read().strip().split(chr(10)))
    print("mem", {k: mi[k].strip() for k in ("MemTotal", "MemAvailable") if k in mi})
except Exception:
    pass'

capture_diagnostics() {
    local b64 mids mid
    b64="$(printf '%s' "$DIAG_PY" | base64 -w0)" || return 0
    mids="$(fly machines list --app "$APP" --json 2>/dev/null \
        | python3 -c 'import sys,json; print(" ".join(m["id"] for m in json.load(sys.stdin) if m.get("config",{}).get("metadata",{}).get("fly_process_group")=="web"))' \
        2>/dev/null)" || return 0
    [ -z "${mids:-}" ] && { diag "no web machines resolved — skipping capture"; return 0; }

    for mid in $mids; do
        diag "--- machine $mid ---"
        timeout "$DIAG_TIMEOUT" fly ssh console --app "$APP" --machine "$mid" \
            -C "python3 -c \"import base64;exec(base64.b64decode('$b64'))\"" \
            >> "$DIAG_LOG" 2>&1 || diag "capture failed for $mid (exit $?)"
    done
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

log "DOWN after $MAX_ATTEMPTS attempts (last http=$code) — capturing diagnostics"
diag "===== DOWN (last http=$code) — pre-restart capture ====="
# Never let evidence-gathering delay or block the recovery it precedes.
capture_diagnostics || log "WARN diagnostics capture failed, continuing to restart"

log "DOWN after $MAX_ATTEMPTS attempts (last http=$code) — restarting $APP"
if fly apps restart "$APP" >> "$LOG" 2>&1; then
    log "Restart issued"
else
    log "ERROR fly apps restart failed (exit $?)"
fi
