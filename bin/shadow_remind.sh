#!/usr/bin/env bash
# Daily on the dev box: score the shadow test (docs/FORECAST_REVIEW_2026_11.md §15) and push one
# ntfy alert when a first week of target days has been scored, and one when the event budget is met.
set -euo pipefail
cd "$(dirname -- "$0")/.."
NTFY_TOPIC="${NTFY_TOPIC:-agile-predict-outage-7eef5967}"  # the topic uptime_monitor.sh alerts on
DIR="${SHADOW_DIR:-logs/shadow}"
[ -e "$DIR/.reminded_budget" ] && exit 0

out="$(.venv/bin/python manage.py shell < analysis/forecast_review_2026_11/shadow_score.py 2>&1)" || {
    echo "$(date -u +%FT%TZ) not scored: $(tail -n 1 <<<"$out")"
    exit 0
}
echo "$(date -u +%FT%TZ)"
echo "$out"

notify() { # $1 marker, $2 title; sent once
    [ -e "$DIR/.reminded_$1" ] && return 0
    grep -E 'target days|event budget|PASS|FAIL' <<<"$out" |
        curl -4 -fsS -H "Title: $2" -H 'Tags: bar_chart' --data-binary @- "https://ntfy.sh/${NTFY_TOPIC}" >/dev/null
    touch "$DIR/.reminded_$1"
}
days="$(sed -n 's/.* \([0-9]*\) target days.*/\1/p' <<<"$out")"
if ! grep -q 'event budget not met' <<<"$out"; then
    notify budget "Shadow test: event budget met, verdict ready"
elif [ "${days:-0}" -ge 7 ]; then
    notify week "Shadow test: first week scored (interim)"
fi
