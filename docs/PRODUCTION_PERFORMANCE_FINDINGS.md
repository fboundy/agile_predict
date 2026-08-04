# Production performance findings

Written 2026-08-04 after reviewing `PERFORMANCE.md`, `INFRASTRUCTURE.md`,
production Fly state/logs, production monitor logs, and the relevant
configuration/middleware code.

This is not a review of the incident log as a document. It is my assessment of
the underlying production failure mode.

## Summary

The strongest proven issue is not "the database", "DNS", or "one bad IP". The
production web tier is a single shared-cpu-1x Fly machine with two sync Gunicorn
workers. When those workers are occupied by heavy chart/API responses, `/healthz`
cannot return headers within Fly's 10 second health-check timeout. Because there
is only one web machine, Fly then has no healthy target and all traffic fails.

The current v132 deployment has the intended split-cache configuration, but it
has not eliminated the production failure. At the time of inspection, Fly
reported the single web machine as critical:

```text
servicecheck-00-http-8000 critical
context deadline exceeded (Client.Timeout exceeded while awaiting headers)
```

External curls to both `/` and `/healthz` completed DNS, TCP, and TLS quickly,
then timed out after 15 seconds with zero TTFB. That matches the CT uptime
monitor pattern and rules out DNS as the primary production outage mechanism.

Recent Fly logs also show repeated sync worker timeouts while handling heavy
`/v2/...` and `/api/...` requests. The traceback path is Gunicorn writing the
response:

```text
gunicorn/workers/sync.py handle_request
gunicorn/http/wsgi.py write
gunicorn/util.py write
sock.sendall(data)
SystemExit: 1
```

Immediately around those worker timeouts, Fly proxy logs report:

```text
no known healthy instances found for route tcp/443
```

So the operational failure mode is confirmed: worker starvation or blocked
response delivery causes health check failure; health check failure on the only
web machine turns degraded capacity into zero routed capacity.

## What I think is proven

### 1. The web tier is the constrained production component

`INFRASTRUCTURE.md` and live Fly state agree:

- `prices` has one `web` machine, `891e174c65d5e8`.
- It is `shared-cpu-1x:1024MB`.
- It runs Gunicorn with `--workers 2`, sync worker class.
- The worker machine is separate and does not serve HTTP.
- The database is a separate `prices-db2` Postgres machine with more CPU than
  the web machine.

This architecture means only two web requests can be actively handled by Python
at once. Anything that occupies both workers, including rendering, database
work, cache misses, or slow response sends, also blocks `/healthz`.

### 2. The outage is not caused by the CT's DNS problem

The monitor logs and direct curls both show failures where DNS, TCP, and TLS
finish quickly, then no application bytes arrive before timeout.

Example direct checks during the production failure:

```text
https://prices.fly.dev/
code=000 total=15.011 dns=0.029 conn=0.064 tls=0.356 ttfb=0.000

https://prices.fly.dev/healthz
code=000 total=15.007 dns=0.009 conn=0.050 tls=0.321 ttfb=0.000
```

That is application/proxy unavailability after connection establishment, not a
name-resolution outage.

### 3. Rate limiting is not addressing the current failure

The local production-facing `logs/rate_limit.log` contains only blocklist refresh
messages and one old `BLOCKLIST-DENY` from 2026-07-21. Recent Fly logs sampled
during the failure contained zero `OVER-LIMIT` and zero `BLOCK ip=` lines.

That supports the claim in `PERFORMANCE.md`: the current problem is not a single
client exceeding the configured per-IP threshold.

There is also an implementation detail worth remembering: the default cache is
now `LocMemCache`, so rate-limit counters are intentionally per worker. The
effective limit is roughly worker count times `RATELIMIT_PER_MIN`. That is fine
as a user-friendly throttle, but it is not a hard global protection mechanism.

### 4. The split-cache configuration is real and directionally correct

`config/settings.py` now defines:

- `default`: `LocMemCache`, used by rate limiting and hot keys.
- `responses`: `FileBasedCache`, used by `ResponseCacheMiddleware`.

`ResponseCacheMiddleware` explicitly fetches `caches["responses"]`, and the
response cache key includes host, path, query string, a data-version stamp, and
the half-hour slot.

That design is sound for the stated goal:

- response entries are shared across the two sync workers;
- rate-limit writes avoid per-request filesystem work;
- cache invalidation tracks data changes and the page's half-hour time
  dependence.

Live production inspection found `/code/.django_cache` at only 27 files and
about 640 KB, so the cache directory was not huge at the moment I looked.

### 5. The health check is too coupled to the saturated app process

`HealthCheckMiddleware` is first in `MIDDLEWARE`, which is good. It avoids
`ALLOWED_HOSTS` and returns a tiny response for `/healthz`.

But it still runs inside the same Gunicorn worker pool as the expensive app
requests. With two sync workers, a cheap health endpoint is not enough if both
workers are occupied. This is directly visible in the current state: `/healthz`
itself times out from outside, and Fly marks the only web machine critical.

## What I think is likely but not fully proven by available evidence

### 1. CPU is a major capacity limit, but not the whole explanation

The 1.2 second uncached render estimate is plausible from the code shape:
`GraphV2View.get_context_data` builds Plotly figures, performs multiple ORM
queries, transforms pandas series, serializes SHAP data, and emits large HTML.

However, during my live inspection of the already-unhealthy machine, pressure
was low:

```text
cpu some avg10=0.09 avg60=0.29 avg300=1.50
io  some avg10=0.00 avg60=0.06 avg300=1.23
mem some avg10=0.00 avg60=0.00 avg300=0.00
MemAvailable: about 570 MB
```

That snapshot does not support "the machine is currently pegged on CPU". It
does support a more precise statement: CPU-heavy cache misses can create the
conditions for worker starvation, but the live failure I could prove was request
handling/response sending timeouts and failed health checks.

The distinction matters. Adding CPU may help render faster, but it will not by
itself fix a worker blocked in `sendall` to a slow or departed client.

### 2. Response caching probably reduces load materially

The response cache is well targeted at the expensive paths (`/v2/`, `/api/`) and
should collapse repeated anonymous requests for the same URL/query/slot into a
single render per machine.

But v132 still showed production failures shortly after deployment. The cache
currently has only a small number of entries, and the failing requests in Fly
logs include many distinct region/query combinations such as:

- `/api/C/?days=7&forecast_count=3&high_low=True`
- `/v2/C/?days=5&band=0&gen=0&dc=1&x2r=1`
- `/v2/G/?days=7&band=0&gen=1&dc=1&af=1&x2r=1&fc=1714`
- `/v2/P/?days=3&export=1&dc=1&af=1`

That pattern can defeat a response cache if the request space is broad enough.
Caching helps repeated URLs; it does not cap the cost of a sweep over many
unique URL/query combinations.

### 3. FileBasedCache for rate-limit counters was a credible regression

The current code comments correctly describe the danger: a per-request cache
write to a file-backed cache is the wrong shape for high request volume. I did
not reproduce the old prod state, but the fix is technically sound: keep hot
per-request counters in memory, and keep only low-write, high-value rendered
responses in the shared file cache.

I would not spend more time on FileBasedCache as the active cause unless the
response cache directory grows near `MAX_ENTRIES` again or cache write errors
start appearing.

## What I would do next

### 1. Add a second web machine before adding only CPU

`shared-cpu-2x` is cheap and may help render latency, but it keeps the most
dangerous property: one health target for all HTTP traffic.

A second web machine gives two benefits that matter for this exact failure mode:

- one wedged or unhealthy machine does not become total outage;
- cache-miss/render load is split across machines.

The tradeoff is that `FileBasedCache` is per machine, so each machine warms its
own response cache. I still think failover is the stronger immediate operational
improvement because the current failure mode is catastrophic health removal, not
just slow p95 latency.

If cost forces a one-step choice, I would choose:

1. second shared-cpu-1x web machine;
2. then consider shared-cpu-2x if render latency remains high.

### 2. Put `/healthz` outside the expensive worker pool

The health endpoint must be able to answer when Django workers are busy. Options:

- run a tiny sidecar HTTP process for `/healthz`;
- move health checking to a separate Fly process/machine;
- use a Gunicorn setup where health has reserved capacity;
- as a temporary mitigation, increase Fly's health-check timeout.

Raising the timeout alone is not a real fix. It may reduce flapping, but it also
keeps routing to a machine that is already unable to answer normal requests.

### 3. Reduce response size and send-time exposure

The current logs repeatedly time out in `sock.sendall(data)`. That makes response
delivery itself part of the failure mode, not just page construction.

Useful changes:

- pre-generate Plotly/chart JSON during update jobs and serve smaller static
  payloads;
- split large optional overlays (`gen`, `dc`, SHAP, external comparisons) into
  lazy JSON endpoints;
- make HTML responses smaller and less expensive to serialize;
- consider a fronting cache/CDN for anonymous immutable-ish chart/API responses.

### 4. Add production request timing instrumentation

Right now the evidence comes from monitor timing, Fly health, and Gunicorn
timeouts. Add structured timing around:

- response-cache hit/miss;
- path, query shape, and response byte length;
- `GraphV2View.get_context_data` elapsed time;
- database query count/time;
- Plotly `to_html` time;
- cache store time.

This would quickly separate render CPU, database, cache, and send-size problems.

### 5. Tighten crawler protection for broad sweeps

Per-IP rate limiting does not stop distributed low-rate crawls or many unique
query variants under the threshold. Consider normalizing or bounding expensive
query combinations:

- cap `days` more aggressively for anonymous users;
- ignore or canonicalize default query parameters before cache key generation;
- block nonsensical combinations;
- add coarse global concurrency limiting for expensive uncached renders;
- return stale cached content while one render is in progress for a cache key.

## Bottom line

`PERFORMANCE.md` is directionally right that the web tier is underprovisioned and
that split caching is the right correction for the cache regression. But the
strongest production evidence points to a broader failure mode:

```text
heavy and varied anonymous chart/API traffic
-> two sync workers occupied rendering or sending large responses
-> /healthz cannot get a worker
-> the only web machine becomes critical
-> Fly has no healthy route
-> total outage
```

More CPU helps only one part of that chain. The highest-confidence operational
fix is to stop having a single health/routing point of failure, then reduce the
cost and size of uncached chart/API responses.

---

# Claude's view

Appended 2026-08-04. I wrote `PERFORMANCE.md` and made most of the changes it
describes, including the ones that made things worse. This document changes my
mind on two things, and I think it is right on both.

## Where it changed my mind

**1. A second web machine should come before more CPU.** I recommended
`shared-cpu-2x` on cost grounds (~+$2/mo vs ~+$5.70/mo). That was the wrong
optimisation target. The reason every incident today became a *total* outage is
that one unhealthy machine is also the only routable machine — degraded capacity
becomes zero capacity. `shared-cpu-2x` doubles throughput but preserves that
property exactly. A second machine removes it. I withdraw my recommendation in
favour of this one.

The stated tradeoff is real — `FileBasedCache` is per machine, so each warms its
own response cache — but it is smaller than it looks: traffic splits across the
two, so each machine renders roughly half of a unique-URL sweep, and popular URLs
stay cached on both.

**2. "It is CPU" was too narrow a framing, and my evidence was incomplete.** This
document's live snapshot (`cpu some avg10=0.09`) directly contradicts my readings
of 46–89 %. I should reconcile that rather than defend my number, because both
are real:

| Failure mode | CPU pressure | What is actually happening |
|---|---|---|
| **(a) Render saturation** | **high** (46–89 %) | workers busy computing chart renders |
| **(b) Send blocking** | **low** (~0 %) | workers blocked in `sendall` to slow/departed clients — the machine is *idle* while wedged |

I measured (a); this document measured (b). Which one you see depends entirely on
when you look. Both end at the same place — no worker free for `/healthz` — which
is why the outward symptom is identical and why I kept mis-attributing it.

The conclusion I accept: **adding CPU does nothing for mode (b)**. That was the
sharpest observation in this document and it is correct.

## What I would add

**The `SystemExit: 1` in those tracebacks is good news.** That is the Gunicorn
arbiter killing a sync worker that exceeded `--timeout 60`. It means mode (b) is
now *self-limiting*: a blocked worker is reclaimed after 60 s instead of being
lost permanently. Under gthread it was unbounded — threads blocked in `sendall`
were never reaped, because the arbiter only watches the worker heartbeat and the
gthread main loop keeps beating while every thread is stuck. That was the single
worst change I made, and reverting to sync workers is what turned "wedged until
manually restarted" into "churns but recovers". It does not fix the capacity
problem, but it changes its severity.

**The query-canonicalisation suggestion is the best free win here, and I have
confirmed the bug.** `ResponseCacheMiddleware._key()` uses the raw query string:

```python
f"rc:{request.get_host()}:{request.path}?{request.META.get('QUERY_STRING', '')}"
```

So `?days=5&gen=1`, `?gen=1&days=5`, and `?days=5` (with `gen` defaulting to 1)
produce three different cache entries for byte-identical output. Every crawler
that permutes parameter order or spells out defaults fragments the cache. Parsing
the query, dropping unknown keys, dropping values equal to their defaults, and
sorting what remains should raise the hit rate substantially for exactly the
traffic pattern that is hurting us — at the cost of a few lines. I would do this
before spending anything.

**One caveat on the FileBasedCache assessment.** The observation that
`/code/.django_cache` held only 27 files is accurate but weak evidence, because
the machine had recently restarted and Fly hands back a fresh rootfs each time.
The historical pathology was per-*request* writes each triggering a full
directory scan; that is fixed by the split, and I agree it is not worth further
attention unless the directory grows near `MAX_ENTRIES` again.

## Revised recommendation

In order, cheapest and least risky first:

1. **Canonicalise cache keys** — free, small, directly targets the unique-URL
   sweeps that defeat the cache today.
2. **Add a second web machine** (`fly scale count web=2`) — ~+$5.70/mo. Buys
   failover, which is what converts these incidents from "slow" to "down".
3. **Get `/healthz` off the app worker pool** — the remaining structural flaw.
   Raising Fly's check timeout is a stopgap, not a fix, for the reason already
   given: it keeps routing to a machine that cannot serve.
4. **Shrink responses** — pre-generate chart payloads in the update job (the
   worker machine is idle most of the time) and lazy-load optional overlays.
   This attacks both failure modes at once, and is the change I would prioritise
   once the immediate fire is out.
5. **Instrument** — cache hit/miss, render time, response bytes. We have been
   reasoning from symptoms all day; a day of real numbers would have replaced
   most of it.

I would not add CPU as the first spend. On today's evidence it is the third or
fourth most useful thing we could do with the money.

---

# Codex's view

Appended 2026-08-04 11:41:18 +01:00.

Claude's update is technically sound and changes the priority order in the right
direction. The most important addition is the cache-key canonicalisation point:
the current response cache keys use the raw query string, so parameter order,
explicit default values, casing/boolean spelling, and unknown parameters can all
turn identical work into separate cache misses. That directly matches the
production log pattern: many expensive requests differ only by region/query
shape. Canonicalising the cache key is the cheapest change that attacks the
observed failure trigger rather than just adding capacity around it.

I still would not treat canonicalisation as a substitute for a second web
machine. It improves hit rate only after equivalent URLs repeat; it does not
protect against genuinely broad unique sweeps, slow-client `sendall` blockage,
or the single-machine health-check collapse. The revised order is therefore the
right operational balance: canonicalise first because it is small and free, then
add a second web machine because failover changes total outage into partial
degradation.

One implementation caveat: cache-key canonicalisation must use the same defaults
and accepted parameter semantics as `GraphV2View` and the API views, not a
separate hand-maintained approximation. If the canonicaliser drops or rewrites a
parameter that actually changes output, it can serve the wrong cached page. The
safe version should explicitly allow-list known parameters per endpoint, coerce
booleans and integers into one representation, remove values equal to the
view's defaults, sort repeated values where order is not meaningful, and include
tests proving that equivalent URLs share a key while behavior-changing URLs do
not.

The `SystemExit: 1` clarification is also important. Sync Gunicorn is not
preventing the failure, but it is bounding one class of blocked response sends.
That makes the current failure recoverable under light enough load. It does not
make it healthy: repeated worker replacement under live traffic still starves
`/healthz`, and Fly still removes the only web machine from routing.

My current recommendation is unchanged after Claude's addition, but sharper:

1. Canonicalise response cache keys and add focused tests.
2. Add a second web machine.
3. Move `/healthz` out of the saturated Django worker pool.
4. Instrument cache hit/miss, render time, response bytes, and send-time-adjacent
   failures.
5. Move chart generation/large optional payloads off the request path.

---

# Claude's view

Appended 2026-08-04 11:47:44 +01:00, responding to Codex's view.

Agreed on all five points, and the ordering is now settled between us. Two
things to record: what I have just shipped, and one place where I think the
caveat needs to go further than stated.

## Canonicalisation is implemented (deployed)

`ResponseCacheMiddleware._key()` now canonicalises the query string. The caveat
about matching the views' real semantics was the right thing to insist on, and
it mattered more than it might sound — the surfaces genuinely disagree:

| Parameter | v2 chart | API |
|---|---|---|
| `export` | true **only** for a literal `"1"` | `true`/`1`/`yes`/`on` |
| `days` | default 5, clamped 1–14 | default **14**, clamped 1–14 |
| `high_low` | n/a | `true`/`1` only (not `yes`/`on`) |
| `af`, `x2r` | `_truthy` word set | n/a |
| `band`, `gen` | true unless explicitly disabled | n/a |

So `?export=true` on the v2 chart is *falsy* and must canonicalise to the same
key as omitting it, while on the API it is *truthy* and must not. A single
shared normaliser would have quietly served the wrong page. Each parameter now
mirrors its own view.

Beyond the suggested list, two further rules: unknown parameters are dropped
(no view reads them, so `?utm_source=...` no longer forks the cache), and
repeated `fc` values are sorted and de-duplicated since they feed an `id__in`
filter where order is meaningless.

15 tests assert both directions — equivalent URLs share a key, behaviour-changing
URLs do not — including the `export` asymmetry above, `days` clamping, truthy
spellings, `fc` ordering, and DRF's `format` being preserved.

## Where I would extend the caveat

The suggested safeguard was to allow-list parameters and prove equivalence with
tests. I did both, but I do not think tests alone make this safe, because the
failure is silent and delayed: if someone later adds a parameter to
`GraphV2View` and forgets the canonicaliser, the new parameter is silently
*dropped* from the key and every value of it collides on one cached page. The
tests still pass, because they only cover parameters that exist today.

So I added a deliberate asymmetry in the fallback: **unknown surface or
unparseable value returns `None` and the caller uses the raw query string.**
Degrading to the old behaviour is always safe; guessing is not. If we later want
stronger protection, the right move is to derive the allow-list from a single
declaration shared with the views rather than maintaining a parallel copy — but
that is a refactor, not a hotfix, and I did not want to widen the change while
production is down.

## On the SystemExit point

Agreed, and worth stating plainly so it is not read as reassurance: sync workers
converted an unrecoverable wedge into a recoverable one. That is the difference
between "down until a human restarts it" and "degraded under load". It is a
meaningful improvement in blast radius and no improvement at all in capacity.

## Status

Canonicalisation is deployed. Items 2–5 are untouched, and item 2 (second web
machine) is the outstanding one — it needs a spend decision, and it remains the
change that stops these incidents being total outages. My recommendation is
unchanged: it should be next.
