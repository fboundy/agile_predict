# Infrastructure

How AgilePredict is hosted and operated: the production app on fly.io, the
self-hosted development/monitoring server, and the jobs and data flows that
connect them. Accurate as of 2026-08-04.

---

## Overview

There are two independent environments with **separate databases**. They are not
replicas of each other — a change to data on one does not appear on the other.

```
                    ┌──────────────────────── fly.io (production) ─────────────────────────┐
  users ──https──▶  │  app "prices"                                                        │
                    │   ├─ web machine    891e174c65d5e8  gunicorn ──┐                      │
                    │   └─ worker machine 801e9df64e56e8  update_worker                     │
                    │                                     │          │                      │
                    │                app "prices-db2"  ◀──┴──────────┘  (Postgres 17)       │
                    └──────────────────────────────────────────────────────────────────────┘
                                 ▲                                  ▲
                    EasyCron ────┘ POST /update                     │ webhook
                    (external)                                 Ko-fi │ POST /webhooks/kofi/
                                                                     │
                    ┌──────────── Proxmox CT "django" (development / ops) ─────────────────┐
                    │  /srv/agile_predict   Django dev server (systemd)  +  SQLite         │
                    │  cron: uptime monitor, watchdog, local updates, backups              │
                    └─────────────────────────────────────────────────────────────────────┘
                                 ▲
                    P:\  ◀── Samba share ── (Windows workstation: editing, git push, fly deploy)
```

---

## Production — fly.io

| | |
|---|---|
| App | `prices` (org `personal`) |
| Region | `lhr` (London) |
| Public URLs | `https://prices.fly.dev`, `https://agilepredict.com` |
| Config | [`fly.toml`](../fly.toml) + [`Dockerfile`](../Dockerfile) |
| Deploy | `fly deploy --app prices` from `P:\` (uses fly's remote builder; no local Docker) |
| Release step | `python manage.py migrate --noinput` runs automatically before each rolling update |

### Machines

| Process | Machine ID | Size | Role |
|---|---|---|---|
| `web` | `891e174c65d5e8` | shared-cpu-1x, 1 GB | gunicorn, serves all HTTP |
| `worker` | `801e9df64e56e8` | shared-cpu-1x, 1 GB | `manage.py update_worker`, runs queued jobs |
| database | `d8d3741b0d7708` (app `prices-db2`) | shared **2 CPU**, 2 GB | Postgres 17 (`flyio/postgres-flex`) |

> Note the web machine has **half the CPU of the database**. The web tier is the
> constrained component — see [PERFORMANCE.md](PERFORMANCE.md).

`auto_stop_machines = 'off'` and `min_machines_running = 1`, so machines never
idle-stop; cost is a fixed per-second rate regardless of utilisation.

`prices-db` (the original database app) is **suspended** — `prices-db2` is live.

### Process commands

```
web    = gunicorn --bind :8000 --workers 2 --timeout 60 --graceful-timeout 30 \
                  --max-requests 800 --max-requests-jitter 200 config.wsgi
worker = python manage.py update_worker
```

Worker class is deliberately **sync**, not `gthread` — threaded workers cannot
recover from threads blocked writing to departed clients. See
[PERFORMANCE.md](PERFORMANCE.md).

### Health check

```
GET /healthz   interval 15s   timeout 10s   grace 30s
```

`/healthz` is answered by `HealthCheckMiddleware` **before** `ALLOWED_HOSTS`
validation, because fly's prober uses the machine's private address as the Host
header and would otherwise be rejected as `DisallowedHost` — which previously
made the check permanently critical and stopped all traffic being routed.

With a single web machine, a failing check means fly has nowhere to route and
**all** traffic fails (`no known healthy instances found for route tcp/443`).

### Secrets (`fly secrets list --app prices`)

`SECRET_KEY`, `ALLOWED_HOSTS`, `DEBUG`, `ENV`, `DATABASE_URL`, `UPDATE_TOKEN`,
`ENTSOE_API_KEY`, `KOFI_VERIFICATION_TOKEN`.

Set with `fly secrets set NAME=value --app prices` (triggers a redeploy).

### Static files

WhiteNoise serves `/static/`; `collectstatic` runs during the image build.

---

## Development / operations server — Proxmox CT `django`

A self-hosted Ubuntu container that doubles as the **monitoring host**. It is not
a staging copy of production — it has its own SQLite database and its own data.

| | |
|---|---|
| Access | `ssh agile@django` |
| OS / Python | Ubuntu 24.04 LTS / Python 3.12.3 (venv at `.venv`) |
| Project root | `/srv/agile_predict` |
| Same tree as | `P:\` on the Windows workstation, via Samba |
| Database | SQLite (`db.sqlite3`) — **not** the production Postgres |
| Disk | ~9.8 GB, ~68 % used |
| Config | `/srv/agile_predict/.env` |

### Web server

Runs as a **systemd user service**, not from a terminal:

```bash
export XDG_RUNTIME_DIR=/run/user/1000
systemctl --user status|restart agile-devserver
```

Unit: `~/.config/systemd/user/agile-devserver.service` —
`manage.py runserver 0.0.0.0:8000 --noreload`, `Restart=on-failure`, logging to
`logs/runserver.log`. **Lingering is enabled** for the `agile` user, so it
survives logout and starts at boot.

> `--noreload` means **code changes require a restart**. Template changes do not.

### Environment variables (`/srv/agile_predict/.env`)

`APP_NAME`, `PGUSER`, `PGPASSWORD`, `DBNAME`, `WIN_BACKUP_FILE`, `UPDATE_TOKEN`,
`ENTSOE_API_KEY`, `FLY_API_TOKEN`, `FLY_DEPLOY_TOKEN`, `FLY_COST_HISTORY`.

### Scheduled jobs (`crontab -l`)

| Schedule | Job | Purpose |
|---|---|---|
| `@reboot` | `manage.py update_worker` | processes queued jobs locally |
| `15 4,10,11,16,22 * * *` | `bin/cron_update.sh` | local forecast update (POSTs to **localhost**) |
| `0 16 * * *` | `bin/cron_latest_agile.sh` | local Agile price refresh |
| `*/5 * * * *` | `bin/uptime_monitor.sh` | polls **production**, logs status + latency, ntfy alerts |
| `45 * * * *` | `bin/watchdog.sh` | polls production; restarts fly machines if down |
| `0 3 * * 1-6` | `bin/incremental_backup.sh` | database backup |
| `0 3 * * 0` | `bin/full_sqlite_backup.sh` | weekly full backup |

`cron_update.sh` / `cron_latest_agile.sh` default to `http://127.0.0.1:8000`, so
they update the **CT's own** database, not production. They authenticate with
`X-Update-Token: $UPDATE_TOKEN`.

### Monitoring

- **`bin/uptime_monitor.sh`** — every 5 minutes, `curl -4` to
  `https://prices.fly.dev/`. Appends `timestamp code total dns= conn= tls= ttfb=`
  to `logs/uptime_monitor.log`, and pushes an ntfy.sh alert on up→down / down→up
  transitions (topic `agile-predict-outage-7eef5967`). The timing breakdown is
  what distinguishes network problems from server problems.
- **`bin/uptime_report.sh [hours]`** — summarises that log into outage episodes
  and an uptime percentage. `bin/uptime_report.sh 24` is the standard check.
- **`bin/watchdog.sh`** — hourly; after 3 consecutive failures runs
  `fly apps restart prices`. It needs `FLY_DEPLOY_TOKEN` (below).
- **`/v2/health/`** — dashboard rendering the above, plus rate-limit offenders.
  Only visible on the CT.
- **GitHub Actions — `prod-update-monitor.yml`** — confirms the EasyCron-triggered
  updates below actually completed. Runs on GitHub-hosted runners (not the CT),
  since prices.fly.dev is not reachable from Claude's cloud agent environment.
  Checks `/api/` (`created_at` of the latest forecast) and `/healthz` shortly
  after each of 06:15/11:15/16:15/22:15 Europe/London; a stale forecast or
  unreachable site fails the job (GitHub's default failure notification) and,
  if the `NTFY_TOPIC` repo secret is set, pushes an alert. See
  `bin/check_prod_update.py`.

### Dev-only pages

`/v2/health/` and `/v2/costs/` appear in the navigation only when
`logs/uptime_monitor.log` exists (the dev-host marker), and `CostsView` also
returns 404 to non-staff elsewhere, so operational and financial data never
appear on the public site.

---

## fly.io API tokens

`flyctl`'s own session token **expires**, which has twice caused silent
failures — including a 6-hour outage where the watchdog detected the problem but
could not act. Long-lived tokens are therefore stored in the CT's `.env`:

| Variable | Scope | Created with | Can it… |
|---|---|---|---|
| `FLY_API_TOKEN` | read-only | `fly tokens create readonly personal --expiry 8760h` | read the Machines API (costs page) — **cannot** restart or SSH |
| `FLY_DEPLOY_TOKEN` | deploy | `fly tokens create deploy --app prices --expiry 8760h` | restart machines (watchdog) |

**Both expire August 2027.** Never source a token from `~/.fly/config.yml` —
that is the short-lived CLI session credential.

Interactive `fly` use from the Windows workstation requires `fly auth login`
periodically.

---

## Data flows

### Forecast updates

**Production** updates are triggered by **EasyCron** (external service) POSTing
to `https://prices.fly.dev/update` roughly every 6 hours (observed at 03:15,
09:15, 15:15, 21:15 UTC). The web process queues an `UpdateJob`; the worker
machine picks it up and runs `manage.py update`. This scheduler is **not in the
repository** — if production forecasts stop while the code is healthy, check
EasyCron first.

**Development** updates come from the CT's own crontab against localhost.

### External comparison forecasts

AgileForecast and X2R are fetched **during update runs**, not per request, and
stored in `ExternalForecast`. The chart reads the newest stored snapshot and
derives other regions from region G. Retention keeps the latest snapshot plus the
first snapshot of each day (for accuracy history).

### Ko-fi income

Ko-fi has no fetch API. Payments arrive by **webhook** at
`https://prices.fly.dev/webhooks/kofi/` (verified against
`KOFI_VERIFICATION_TOKEN`) into production's database. Historic data was imported
into **both** environments with `manage.py import_kofi_csv <export.csv>`
(idempotent on transaction id). The dev costs page reads production totals via
token-authenticated `GET /webhooks/kofi/summary`, preferring production only when
it holds at least as many payments as the local table, so history cannot vanish.

---

## Caching

Two caches with deliberately different backends:

| Alias | Backend | Used for | Why |
|---|---|---|---|
| `default` | LocMemCache | rate-limit counters, blocklist state | written on **every** request; must be O(1) with no I/O |
| `responses` | FileBasedCache (`CACHE_DIR`) | rendered pages/API responses | must be **shared across workers**, else each worker re-renders every URL |

Response cache keys are **event-keyed**: they embed a data-version stamp (latest
forecast id + latest price timestamp) and the current 30-minute slot, so entries
are reused until the content actually changes rather than expiring on a timer.

> Never point a per-request cache at `FileBasedCache` — it re-lists its entire
> directory on every write, which wedged production. See
> [PERFORMANCE.md](PERFORMANCE.md).

---

## Request-path protection

Applied in `MIDDLEWARE` order: `HealthCheckMiddleware` → `BlocklistMiddleware` →
`RateLimitMiddleware` → … → `ResponseCacheMiddleware`.

- **Blocklist** — FireHOL level 1 (~3,900 IPv4 ranges), refreshed in a background
  thread, cached to `CACHE_DIR/blocklist.pickle`. Non-global addresses are never
  blocked, so LAN/localhost access is safe. `manage.py update_blocklist`.
- **Rate limit** — 60 requests/minute per IP (`RATELIMIT_PER_MIN`), escalating to
  a 600 s block for repeat offenders. Reads the real client IP from
  `Fly-Client-IP`. Fail-open. Counters are per-worker, so the effective limit is
  roughly 2×.
- **robots.txt** — disallows bulk AI crawlers (GPTBot, ClaudeBot, CCBot,
  PerplexityBot and similar), which were confirmed sweeping every
  region × filter combination.

Limits are documented publicly at `/v2/limitations/`.

---

## Common operations

```bash
# Deploy (from P:\ on Windows)
fly deploy --app prices

# Production status / logs / shell
fly status --app prices
fly logs --app prices --no-tail
fly ssh console --app prices -C "sh -c 'head -1 /proc/pressure/cpu'"

# Restart production
fly machine restart <machine-id> --app prices

# Dev server
ssh agile@django
export XDG_RUNTIME_DIR=/run/user/1000 && systemctl --user restart agile-devserver

# Uptime report
ssh agile@django 'cd /srv/agile_predict && bash bin/uptime_report.sh 24'
```

### Gotchas

- **Samba strips executable bits.** A `git checkout`/`reset` performed from `P:\`
  removes `+x` from `bin/*.sh`, silently disabling cron jobs (`Permission
  denied`). They are tracked as mode 755, but after any branch operation run
  `ssh agile@django 'chmod +x /srv/agile_predict/bin/*.sh'`. Prefer running git
  branch operations **on the CT**.
- **The CT has no GitHub credentials.** Commit there if convenient, but `git
  push` from `P:\`.
- **`ps`, `free` and `ss` are not installed** in the fly image — use `/proc`, and
  prefer Python over awk (quoting breaks through `fly ssh -C`).
- **The CT's router mishandles AAAA lookups**, adding ~5 s to sparse DNS queries.
  Scripts use `curl -4`; this still occurs occasionally and inflates measured
  latency without affecting production.
- **`fly ssh console` supports stdin**, which is the easiest way to copy a file
  to production:
  `cat f | fly ssh console --app prices -C "sh -c 'cat > /tmp/f'"`.
