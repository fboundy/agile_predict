#!/usr/bin/env bash
# uptime_report.sh — summarize logs/uptime_monitor.log into outage episodes.
# Usage: bin/uptime_report.sh [hours]   (default: all history)

set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LOG="${SCRIPT_DIR}/../logs/uptime_monitor.log"
HOURS="${1:-}"

if [ -n "$HOURS" ]; then
    CUTOFF=$(date -u -d "-${HOURS} hours" '+%Y-%m-%dT%H:%M:%SZ')
    DATA=$(awk -v cutoff="$CUTOFF" '$1 >= cutoff' "$LOG")
else
    DATA=$(cat "$LOG")
fi

echo "$DATA" | awk '
{
    total++
    code = $2
    ok = (code ~ /^[23]/)
    if (ok) { ok_count++ } else { fail_count++ }

    if (!ok && !in_episode) {
        in_episode = 1
        episode_start = $1
        episode_first_code = code
    }
    if (ok && in_episode) {
        in_episode = 0
        episodes++
        printf "  Outage: %s -> %s (last_code=%s)\n", episode_start, prev_ts, prev_code
    }
    prev_ts = $1
    prev_code = code
}
END {
    if (in_episode) {
        episodes++
        printf "  Outage: %s -> ONGOING (last_code=%s)\n", episode_start, prev_code
    }
    print "---"
    printf "Total checks: %d  OK: %d  Failed: %d  Uptime: %.2f%%\n", total, ok_count, fail_count, (total > 0 ? 100*ok_count/total : 0)
    printf "Distinct outage episodes: %d\n", episodes
}
'
