#!/usr/bin/env python3
"""Confirm production forecast updates ran at 06:15/11:15/16:15/22:15 Europe/London.

Scheduled via .github/workflows/prod-update-monitor.yml on GitHub-hosted
runners, because the production site is not reachable from Claude's cloud
agent environment (org egress policy). The workflow fires every 15 minutes
inside UTC windows that bracket each expected run, covering both BST and GMT
offsets; this script then uses the real Europe/London clock to decide whether
`now` is actually inside a check window, and no-ops otherwise so it only acts
once per real event regardless of DST.

Exits non-zero (failing the GitHub Actions job, which triggers GitHub's
default failure notification) when the latest forecast wasn't refreshed
after the expected run, or the site is unreachable. Set the NTFY_TOPIC repo
secret to also push an alert via ntfy.sh, matching bin/uptime_monitor.sh.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

LONDON = ZoneInfo("Europe/London")
EXPECTED_TIMES = [(6, 15), (11, 15), (16, 15), (22, 15)]
BUFFER_START = timedelta(minutes=20)  # give the worker time to pick up and run the job
BUFFER_END = timedelta(minutes=50)
API_URL = "https://prices.fly.dev/api/"
HEALTHZ_URL = "https://prices.fly.dev/healthz"
TIMEOUT = 20


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "agile-predict-update-monitor"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.status, resp.read()


def current_window(now):
    """Return the expected run datetime if `now` falls inside its check window."""
    for hour, minute in EXPECTED_TIMES:
        expected = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if expected + BUFFER_START <= now <= expected + BUFFER_END:
            return expected
    return None


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
    expected = current_window(now)
    if expected is None:
        print(f"{now.isoformat()} outside a check window, nothing to do")
        return 0

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
