#!/usr/bin/env python3
"""Confirm production forecast updates ran at 06:15/11:15/16:15/22:15 Europe/London.

Scheduled via .github/workflows/prod-update-monitor.yml on GitHub-hosted
runners, because the production site is not reachable from Claude's cloud
agent environment (org egress policy). GitHub Actions schedules on
low-activity repos are best-effort and run at times only loosely related to
the configured cron (observed: a schedule targeting 8 fixed ticks/day still
fired ~4/day at essentially arbitrary times). So this script does not assume
it runs anywhere near a particular expected time: whenever it runs, it finds
the most recent of the four daily expected runs that has already passed
(with a grace buffer for the job to complete) and checks the latest forecast
is no older than that. That makes correctness independent of when GitHub
actually decides to fire the job, as long as it fires at least once between
consecutive expected runs (~4-6 hours apart).

Exits non-zero (failing the GitHub Actions job, which triggers GitHub's
default failure notification) when the latest forecast wasn't refreshed
after the most recent expected run, or the site is unreachable. Set the
NTFY_TOPIC repo secret to also push an alert via ntfy.sh, matching
bin/uptime_monitor.sh.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
UTC = ZoneInfo("UTC")
EXPECTED_TIMES = [(6, 15), (11, 15), (16, 15), (22, 15)]
# EasyCron was last verified firing at these UTC times (= the London times above in
# BST). It is not recorded whether its schedule is set in UTC or Europe/London, so
# in GMT the runs may come an hour earlier than EXPECTED_TIMES. Checking against
# the earlier of the two readings avoids four false alarms a day all winter if it
# is UTC, at the cost of noticing a missed run up to one slot later.
EXPECTED_TIMES_UTC = [(5, 15), (10, 15), (15, 15), (21, 15)]
GRACE = timedelta(minutes=20)  # give the worker time to pick up and run the job
API_URL = "https://prices.fly.dev/api/"
HEALTHZ_URL = "https://prices.fly.dev/healthz"
TIMEOUT = 20


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "agile-predict-update-monitor"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.status, resp.read()


def _most_recent(now, times, tz):
    local_now = now.astimezone(tz)
    candidates = []
    for day_offset in (0, -1):
        day = local_now + timedelta(days=day_offset)
        for hour, minute in times:
            candidates.append(day.replace(hour=hour, minute=minute, second=0, microsecond=0))
    past = [c for c in candidates if c + GRACE <= local_now]
    return max(past).astimezone(LONDON)


def most_recent_expected(now):
    """Return the latest expected run datetime that is already past its grace period,
    reading the schedule as London time or UTC, whichever is earlier."""
    return min(_most_recent(now, EXPECTED_TIMES, LONDON), _most_recent(now, EXPECTED_TIMES_UTC, UTC))


def send_ntfy_alert(message):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        return
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=message.encode(),
            headers={
                "Title": "AgilePredict prod update check failed",
                "Priority": "urgent",
                "Tags": "rotating_light",
            },
        )
        urllib.request.urlopen(req, timeout=TIMEOUT)
    except Exception as exc:
        print(f"WARN: ntfy alert failed: {exc}")


def main():
    now = datetime.now(LONDON)
    expected = most_recent_expected(now)

    print(f"Checking update expected at {expected.isoformat()} (now {now.isoformat()})")
    problems = []

    try:
        status, _ = fetch(HEALTHZ_URL)
        if status != 200:
            problems.append(f"/healthz returned HTTP {status}")
    except Exception as exc:
        problems.append(f"/healthz request failed: {exc}")

    try:
        status, body = fetch(API_URL)
        if status != 200:
            problems.append(f"/api/ returned HTTP {status}")
        else:
            data = json.loads(body)
            if not data:
                problems.append("/api/ returned no forecasts")
            else:
                created_at = datetime.fromisoformat(data[0]["created_at"].replace("Z", "+00:00"))
                created_at_london = created_at.astimezone(LONDON)
                if created_at_london < expected:
                    age = now - created_at_london
                    problems.append(
                        f"latest forecast created_at={created_at_london.isoformat()} "
                        f"predates the {expected.strftime('%H:%M')} run (stale by {age})"
                    )
                else:
                    print(f"OK: latest forecast created_at={created_at_london.isoformat()}")
    except Exception as exc:
        problems.append(f"/api/ request failed: {exc}")

    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        send_ntfy_alert(f"Update expected {expected.strftime('%H:%M')} London: " + "; ".join(problems))
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
