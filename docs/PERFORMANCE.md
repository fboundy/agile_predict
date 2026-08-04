# Production performance / availability — investigation log

Running record of the availability problems on the fly.io app `prices`, what has
been tried, and what is still outstanding. Newest information at the top of each
section. Written 2026-08-04.

## Current status

**Recovering.** Production was intermittently down through the morning of
2026-08-04. The most recent change (splitting the caches — shared file cache for
rendered responses, in-memory for rate-limit counters) produced an immediate and
large improvement:

| Signal | While wedged | After split-cache deploy (v132) |
|---|---|---|
| CPU pressure (`/proc/pressure/cpu`) | **46–89 %** | **5.9 %** |
| Response time | timeout at 15 s | 0.15 s |
| fly health check | critical | passing |
| Shared response cache in use | no (per-process) | yes (59 entries) |

This is consistent with the diagnosis: making the response cache per-process
multiplied render cost by the worker count, and restoring a shared cache means a
URL is rendered once per machine instead of once per worker.

**Caveat: this needs a full day to confirm.** Several earlier fixes looked good
for an hour and then degraded, so treat the above as encouraging rather than
settled. The measure that matters is `bin/uptime_report.sh 24`.

The underlying arithmetic has not changed: an uncached v2 chart render costs
roughly **1.2 s of Python/Plotly work** and the machine has **one shared vCPU**,
so sustained cache-miss traffic can still saturate it. When that happens
`/healthz` cannot be answered within its 10 s timeout, fly pulls the machine from
rotation, and *everything* fails until load drops — partial capacity becomes zero
capacity.

Uptime by configuration (from `logs/uptime_monitor.log`, 5-minute polls):

| Configuration | Uptime |
|---|---|
| Baseline (sync workers, no cache) | 74.8 % |
| + threaded (gthread) workers | 75.4 % |
| + event-keyed response cache | **91.0 %** |
| + persistent DB connections | 55 % → total outage |
| Currently (post-revert, split caches) | being measured |

## What has been tried

### Helped
- **Response caching of anonymous GETs** (`ResponseCacheMiddleware`) — the single
  biggest win, 75 % → 91 %. Cache hits skip both the render and the database.
- **Event-keyed cache invalidation** — keys embed a data-version stamp (latest
  forecast id + latest price timestamp) and the current 30-minute slot, so
  entries live until the content actually changes instead of expiring on a timer.
  Better hit rate *and* fresher than any fixed TTL.
- **Serving external comparison forecasts from stored data** instead of live
  AgileForecast/X2R calls — removed 5–15 s blocking upstream calls from the
  request path.
- **`robots.txt`** disallowing bulk AI crawlers (GPTBot et al.), which were
  confirmed sweeping every region × filter combination.
- **Per-IP rate limiting** and a **reputation IP blocklist** — these do stop
  abusive single sources, but measurement showed *no* IP near the 60/min limit
  during the outages, so they are not addressing the current problem.

### Did not help
- **Threaded (gthread) gunicorn workers.** No measurable improvement (75.4 %),
  and actively harmful — see below. Reverted to sync workers.
- **Persistent database connections (`CONN_MAX_AGE=60`).** Deployed on the theory
  that per-request connection setup was the bottleneck. Uptime fell 91 % → 55 %
  and ended in a 6-hour outage. Reverted. Causation was never proven and is now
  considered unlikely — the machine kept wedging after the revert.
- **`LocMemCache` for everything.** Fixed one real problem (below) but created a
  worse one: the response cache became per-process, so every worker rendered
  every URL independently. With 3 workers that tripled CPU load.

### Made things worse (own goals, since corrected)
- **gthread + slow clients = permanent wedge.** Captured live on a stuck machine:
  all 16 threads (2 workers × 8) blocked in `wait_woken` on sockets, 16 sockets
  holding >10 KB in their send queue, and **43 of 45 connections in
  `CLOSE_WAIT`** — clients had disappeared mid-response while threads blocked
  forever in `sendall` writing large chart pages. gthread cannot recover from
  this: gunicorn's `--timeout` only fires when a worker's *heartbeat* stops, and
  the gthread main loop keeps beating while every one of its threads is blocked,
  so the arbiter never intervenes. Sync workers self-heal, because the arbiter
  kills a worker whose request exceeds `--timeout`. **Reverted to sync workers.**
- **`FileBasedCache` used for per-request writes.** Django's `FileBasedCache`
  re-lists its entire directory on every write to decide whether to cull, and
  `RateLimitMiddleware` writes on *every* request — so each request performed a
  directory scan growing toward `MAX_ENTRIES` (plus a 250-file delete once over).
  This produced gradual degradation as the directory filled and temporary
  recovery after a restart (fly hands back a fresh rootfs, emptying it).
  **Now split:** `LocMemCache` for rate-limit counters (per-request, O(1), no
  I/O) and a shared `FileBasedCache` for rendered responses (written only on a
  miss, and shared so a URL is rendered once per machine rather than once per
  worker).

### Fixed along the way (not performance, but was hiding the problem)
- **The watchdog was dead.** `bin/watchdog.sh` correctly detected outages and ran
  `fly apps restart`, but flyctl had no credentials under cron
  (`no access token available`) because the CLI session token expires. This is
  why the 2026-08-04 outage ran ~6 hours unattended. It now reads a long-lived
  deploy-scoped token (`FLY_DEPLOY_TOKEN`, expires Aug 2027) from `.env`.
  **Read-only fly tokens cannot restart machines or SSH — deploy scope is
  required.**
- **CT-side DNS stalls.** The monitoring host's router mishandles AAAA lookups,
  adding 5 s to roughly 10 % of checks. `curl -4` reduced but did not eliminate
  it. This inflates *measured* latency only; every real failure showed fast DNS.

## Ruled out

| Hypothesis | Evidence against |
|---|---|
| Memory / OOM | Memory pressure 0.00, ~500 MB available, no OOM kills, no restarts |
| Disk I/O | I/O pressure 0.00–0.18 % during wedges |
| Database saturation | `SELECT 1` in 1 ms, 11 of 300 connections in use, DB app health 3/3 passing |
| Crawler / single abusive IP | Only ~6 rate-limit log lines; no IP near the 60/min cap |
| Cache stampede after updates | Forecasts land 03:15 / 09:16 / 15:16 UTC; failure bursts at 08:20 / 12:30 / 16:15 — no correlation |
| Slow upstream APIs | External forecasts now served from stored data; no live calls on the request path |

## Suggested next steps

**1. Add CPU — this is the evidence-backed fix.** Every software optimisation
worth doing has been done, and the remaining constraint is arithmetic: ~1.2 s of
CPU per uncached render against one shared vCPU. Options, cheapest first:

| Option | Command | Approx. cost |
|---|---|---|
| **Double CPU on the same machine** (recommended) | `fly scale vm shared-cpu-2x --app prices` | **~+$2/mo** |
| Second web machine (adds failover too) | `fly scale count web=2 --app prices` | ~+$5.70/mo |
| Dedicated core | `fly scale vm performance-1x --app prices` | ~+$25/mo |

`shared-cpu-2x` is the best value: it doubles capacity for roughly £1.50/month,
keeps a single shared response cache (a second machine would need its own), and
is a one-command change that is trivially reversible. The costs page shows Ko-fi
income covering hosting with ~£75 to spare over 20 months, so this is affordable.

**2. Reduce the cost of a render.** Independent of scaling, and free:
- Profile `GraphV2View.get_context_data` — 1.2 s is a lot for one chart. Likely
  candidates are Plotly figure construction and the per-slot SHAP payload.
- Consider generating the chart JSON once per forecast (in the update job, on the
  worker machine) and serving it statically, rather than rendering per request.
  This would move nearly all chart CPU off the web machine entirely and is
  probably the highest-leverage change available.

**3. Protect `/healthz`.** The current failure mode is catastrophic rather than
graceful: once workers are busy, the health check fails and fly drops *all*
traffic, so partial capacity becomes zero capacity. Options: raise the check
`timeout` (currently 10 s) while on a single machine, or serve the health check
from a separate lightweight process.

**4. Keep watching the split-cache change.** The most recent deploy separates
rate-limit counters (in-memory) from the shared response cache. If uptime returns
to ~91 % it confirms the LocMemCache regression is resolved; if the machine still
saturates, that is further evidence for step 1.

## How to diagnose a wedge (this worked; repeat it)

`fly ssh console` still works when HTTP is dead. `ps`, `free` and `ss` are **not**
in the image — use `/proc`, and prefer Python over awk (quoting breaks through
`fly ssh -C`):

```bash
# Is it CPU, disk or memory?
cat /proc/pressure/cpu /proc/pressure/io /proc/pressure/memory

# What are the threads doing?
for t in /proc/*/task/*/wchan; do cat $t; echo; done | sort | uniq -c | sort -rn

# Are responses stuck in socket send queues? (port 8000, parse in python)
#   tx_queue > 0 with state 08 (CLOSE_WAIT) == blocked writing to departed clients
```

The uptime monitor also records curl's `dns/conn/tls/ttfb` breakdown per check,
which is what distinguished "the CT's DNS is slow" from "the server accepted the
connection and never answered".
