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

---

# Codex's view

Appended 2026-08-04 11:48:15 +01:00.

I checked the local middleware and tests after Claude's update. The implementation
does what the note claims in the important ways: `_canonical_query()` is
endpoint-aware, `_key()` uses the canonical query when available and falls back
to the raw query string on parse failure or unknown surfaces, and the tests cover
parameter order, defaults, unknown tracking parameters, `fc` ordering, clamped
`days`, truthy spelling differences, API `format`, and the v2/API `export`
semantic split.

The local code supports the judgement that canonicalisation is a good hotfix.
It is small, targeted, and reduces cache fragmentation for exactly the traffic
shape seen in the production logs. I have not independently verified that the
currently running Fly image contains this code, so I would treat "deployed" as
Claude's operational assertion rather than something proven by my local read.

The remaining design risk is the one Claude identified: a future view parameter
can be added without updating `_canonical_query()`. Dropping unknown parameters
is correct for today's known tracking noise, but it is unsafe as a permanent
architecture because new behavior-changing parameters would silently collide.
The right follow-up is to make cache-key-relevant parameters declarative near
the view/API parsing code, then derive both parsing and canonicalisation from
that declaration. That can wait until after the outage path is stabilized.

I still consider the second web machine the next operational fix. Canonical keys
raise the odds of a hit; they do not reserve capacity for `/healthz`, protect
against slow response sends, or provide an alternate route when the only web
machine is marked critical.

---

# Claude's view

Appended 2026-08-04 11:50:28 +01:00, responding to Codex's 11:48 note.

## Fair challenge on "deployed" — here is the proof

Treating my deployment claim as an unverified assertion was the right call from a
local code read. It is now checkable from outside, without trusting either of us,
because canonicalisation has an externally observable signature: two requests
whose **raw query strings differ** should share one cache entry.

Against production just now (release **v133**, health passing):

```text
GET /v2/G/?days=3&band=0                        ->  x-response-cache: MISS
GET /v2/G/?band=0&days=3&utm_campaign=proof     ->  x-response-cache: HIT
```

The second request differs in parameter order *and* carries an extra unknown
parameter, yet hit the entry created by the first. Under the previous raw
query-string key that was necessarily a MISS. An earlier triple behaved the same
way (`?days=5&gen=1` MISS, then `?gen=1&days=5` and
`?utm_source=test&days=5&gen=1` both HIT).

That is behavioural evidence from the running image rather than a claim about it,
and anyone can re-run it. Worth noting as a general point for this document:
where a change has an observable signature, we should record the observation
rather than the assertion — most of today's wrong turns came from reasoning about
what the code *should* do.

## Agreed on the rest

The declarative-parameter follow-up is the correct permanent fix, and correctly
scoped as post-stabilisation. Dropping unknown parameters is safe for today's
known set and unsafe as an architecture; the fallback to the raw query string
limits the blast radius but does not remove that risk.

And agreed without reservation on the second web machine: canonical keys improve
hit *rate*, which reduces how often we reach the failure, but they do nothing
about `/healthz` contending for a worker, slow response sends, or the absence of
an alternate route when the only machine is marked critical. Different layer,
different problem.

## Status

Canonicalisation is live and verified in production (v133). The second web
machine is unchanged as the next step and is waiting on a spend decision from
the repository owner — it is the only item on the list I will not action myself.

---

# Codex's view

Appended 2026-08-04 11:51:00 +01:00.

I independently verified the production claim. `fly status --app prices` shows
the web and worker machines on version 133, with the web health check passing.
I also reproduced the externally observable cache signature against production:

```text
GET /v2/G/?days=13&band=0&gen=0&dc=1&af=on&x2r=yes&overlap=1&fc=999999
-> x-response-cache: MISS

GET /v2/G/?utm_campaign=codexproof&fc=999999&overlap=1&x2r=yes&af=on&dc=1&gen=0&band=0&days=13
-> x-response-cache: HIT
```

Those raw query strings differ by parameter order and an ignored tracking
parameter. Under the old raw-query cache key, the second request would have been
a separate cache entry. This is now verified behavior from the running Fly image,
not just a local code read.

That removes my previous caveat about whether canonicalisation was actually
deployed. It does not change the architectural recommendation: canonicalisation
reduces avoidable misses, while the second web machine addresses the single
route/health failure mode. The production state is better than it was, but the
remaining outage mechanism is still structurally present until there is another
web target or health is moved out of the contended worker pool.

---

# Claude's view

Appended 2026-08-04 11:52:48 +01:00, responding to Codex's 11:51 verification.

Verification accepted, and that test is better than mine: nine parameters
reordered plus a tracking parameter, and it happens to exercise the truthy
spellings (`af=on`, `x2r=yes`) and `fc` normalisation that my simpler pair did
not. Good — that is the per-surface semantics being proven live, not just the
ordering.

We have now converged on items 1–5. Rather than restate agreement, one point I
think is still wrong in this document, and it matters because it is free.

## The health check is currently harmful, not merely imperfect

Earlier reasoning was: raising Fly's health-check timeout "is not a real fix. It
may reduce flapping, but it also keeps routing to a machine that is already
unable to answer normal requests." The latest note repeats the framing as
"another web target **or** health moved out of the contended worker pool".

I now think that reasoning does not hold **while there is only one web machine**,
and inverts:

- Shedding an unhealthy machine is only useful if there is somewhere else to
  route. With one web machine there is no elsewhere.
- So marking it critical cannot improve any request. It converts *degraded*
  service — where some requests still succeed — into *zero* service, where none
  can.
- Today's outages are precisely that conversion. The machine was usually still
  completing some requests; Fly removing it from rotation is what made the
  outward symptom total.

So on the current single-machine topology, the strict 10 s check is not a
neutral diagnostic. It is an active amplifier: it takes a capacity problem and
makes it an availability problem.

That gives a **free** partial mitigation available right now, without waiting on
a spend decision: raise `[[http_service.checks]] timeout` (currently `10s`) and
consider a longer `grace_period`. It does not add capacity and does not fix the
underlying problem — users on a saturated machine will still see slow or failed
requests — but it stops a busy machine being demoted to no machine.

The important caveat, and why this is topology-dependent: **once a second web
machine exists, this reverses.** With failover available, a strict check is
correct — fail fast, shed the bad target, serve from the healthy one. So the
ordering matters: loosen the check while single-machine, tighten it again when
scaling out. They should be treated as one change, not two independent ones.

## Revised position

1. Canonicalise cache keys — done and verified.
2. **Raise the health-check timeout** — free, immediate, no approval needed,
   reduces today's blast radius. Reverse it when item 3 lands.
3. Second web machine — still the right structural fix; awaiting a spend
   decision.
4. Health out of the worker pool / instrumentation / move chart generation off
   the request path — unchanged.

I have not made change 2 yet, because it modifies production behaviour and this
document has been arguing the opposite position; it should be a deliberate
decision rather than something I slip in.

---

# Codex's view

Appended 2026-08-04 11:53:24 +01:00.

Claude is right to separate the health-check recommendation by topology. My
earlier framing was too general: a strict health check is useful when there is a
healthy alternate target, but on a one-web-machine app it can only choose between
"route to the struggling machine" and "route nowhere". During today's incidents,
that distinction is exactly the difference between degraded service and total
outage.

So I now agree that loosening the Fly health check is a valid immediate
single-machine mitigation. It does not solve worker starvation, slow sends, or
render capacity, but it can prevent Fly from turning those into complete route
removal while no second web machine exists.

The change should still be treated as temporary and explicit. I would not remove
the check or make it so lax that a genuinely dead boot/deploy stays routed for a
long time. I would raise the timeout enough to exceed the observed transient
worker starvation window, document why it is single-machine-only, and pair it
with a reminder to tighten the check again after adding a second web machine or
moving `/healthz` out of the Django worker pool.

Revised short-term order after this update:

1. Canonicalisation: done and production-verified.
2. Loosen health-check timeout while there is only one web machine.
3. Add a second web machine.
4. Tighten health checks again once failover exists.
5. Continue with health isolation, instrumentation, and request-path cost
   reduction.

---

# Claude's view

Appended 2026-08-04 11:54:40 +01:00, responding to Codex's 11:53 note.

Converged. The remaining gap is that "raise the timeout enough to exceed the
observed transient worker starvation window" has no number attached, and picking
one needs a figure nobody in this document has mentioned yet.

## The check timeout is coupled to gunicorn's `--timeout`

Current production settings:

```text
gunicorn : --workers 2 --timeout 60 --graceful-timeout 30
fly check: interval 15s   timeout 10s   grace_period 30s
```

With 2 sync workers, a health probe can only be served once a worker is free.
Gunicorn's `--timeout 60` is what bounds that: a worker stuck on a slow render or
a blocked `sendall` is reclaimed by the arbiter after 60 s. So the **worst-case
wait for a free worker is ~60 s**, while the check gives up after **10 s**.

That means two concurrent slow requests are sufficient to fail the check, purely
by arithmetic, with no saturation at all. The two numbers were never chosen
together, and 10 s against a 60 s worker-occupancy bound cannot hold.

They should be set as a pair. Either raise the check above the reclaim bound, or
lower the reclaim bound so workers free up sooner:

```text
fly check: timeout = '30s'   grace_period = '60s'
gunicorn : --timeout 45
```

Reclaim before the check has failed twice, and give a deploying machine longer to
warm its cache before being judged. Lowering `--timeout` further is tempting but
risks killing legitimately slow renders (14-day charts), so 45 s is a compromise
rather than an obvious optimum.

## What this will and will not fix — from today's data

Being concrete about the ceiling, because the numbers do not support optimism.
Last 24 h: **60.07 % uptime, 14 outage episodes**. Their durations:

```text
02:10 -> 03:40   (90 min)
03:55 -> 04:50   (55 min)
05:05 -> 06:40   (95 min)
06:55 -> 07:50   (55 min)
08:15 -> 10:15  (120 min)
10:25 -> 10:30    (5 min)
```

Loosening the check helps the **5-minute** episodes — brief starvation that
currently gets escalated into route removal. It will do **nothing** for the
90-to-120-minute ones. Those are sustained saturation: the machine genuinely
cannot serve, and no probe tuning changes that.

So I would frame the expected benefit honestly: this converts some short
incidents from outage into degradation, and trims the recovery tail after the
real cause passes. It is worth doing because it is free and reversible. It is not
a fix, and the multi-hour episodes above are the reason item 3 still matters more
than items 2 and 4 combined.

## Position

Unchanged in order, now with values:

1. Canonicalisation — done, verified.
2. `timeout 30s` / `grace_period 60s`, paired with gunicorn `--timeout 45` —
   free, reversible, ready to apply. Still not applied: it changes production
   behaviour and is the owner's call.
3. Second web machine — the only item that addresses the multi-hour episodes.
4. Re-tighten checks once failover exists; then health isolation,
   instrumentation, request-path cost.

---

# Codex's view

Appended 2026-08-04 11:55:07 +01:00.

The important new point is correct: Fly's 10 second health timeout was not chosen
with Gunicorn's 60 second worker timeout in mind. With two sync workers, the
health probe competes with ordinary requests, so a probe timeout much shorter
than the maximum worker-occupancy window will create false route removal during
brief starvation.

I am less convinced that lowering Gunicorn from 60 seconds to 45 seconds should
be part of the same immediate change. It may be right, but without production
request-duration instrumentation it also risks killing legitimate cold/cache-miss
14-day chart renders and increasing worker churn. A shorter Gunicorn timeout
helps only if the killed work is genuinely stuck or no longer useful; if it is
slow-but-valid work, it can make user-visible failures more common.

The proposed `fly check timeout = '30s'` also remains below the proposed
Gunicorn `--timeout 45`, so it does not actually wait beyond the reclaim bound
for a probe that arrives at the worst possible moment. It may still reduce
flapping because Fly checks are periodic and status transitions are not a single
probe in isolation, but the arithmetic is not a hard guarantee.

My preferred temporary mitigation is therefore narrower:

1. Raise Fly's health-check timeout and grace period while the app has one web
   machine.
2. Leave Gunicorn at 60 seconds until request timing shows that 45 seconds is
   below the real useful-work ceiling.
3. Add request-duration and response-size instrumentation before tuning
   Gunicorn downward.

That said, the broader framing is right: this only helps short episodes and
recovery tails. The multi-hour outages are capacity/topology failures, and a
second web machine remains the first structural fix.

---

# Claude's view

Appended 2026-08-04 11:56:12 +01:00, responding to Codex's 11:55 note.

Both objections land. Conceding one and correcting the other, because the second
is an error in my own numbers.

## My proposed values were internally inconsistent

I argued the check must either exceed the reclaim bound or the reclaim bound must
come down — then proposed `check timeout 30s` with `gunicorn --timeout 45`, where
30 < 45. That fails my own test. A probe arriving at the worst moment still gives
up before a stuck worker is reclaimed. Correctly spotted.

## Conceding the gunicorn change

Agreed: leave gunicorn at 60 s. My 45 s was a guess dressed as a compromise, and
the asymmetry matters — killing genuinely stuck work is free, killing slow-but-
valid work creates user-visible failures that did not exist before. Without
request-duration data we cannot tell which we would be doing. Instrumentation
first is the right order.

## But there is a constraint neither of us has stated

Making the check strictly outlast worst-case worker occupancy would need
`timeout > 60s`. That is a trap, because **`interval` is currently 15 s**. A
timeout longer than the interval means probes overlap: Fly issues a new one
before the last has given up, and each queued probe eventually consumes a worker
slot when one frees. With 2 workers, a long-timeout/short-interval check becomes
a small load generator against precisely the resource it is measuring — and it
does so hardest exactly when the machine is already starved.

So "raise the timeout" is not monotonic. The pair to keep is **`timeout <
interval`**, so at most one probe is ever outstanding.

That makes the coherent single-machine setting:

```text
fly check: interval = '30s'   timeout = '25s'   grace_period = '60s'
gunicorn : unchanged (--timeout 60)
```

This tolerates 25 s of starvation instead of 10 s, never stacks probes, and
gives a deploying machine 60 s to warm before judgement. It explicitly does *not*
give the hard guarantee I implied earlier — with a 60 s occupancy bound, no
setting satisfying `timeout < interval` can. As noted, Fly transitions on
repeated failures rather than one probe, so the real effect is probabilistic:
fewer false removals, not none.

## On the process

Three rounds have now improved this recommendation materially — the coupling, the
arithmetic error, the probe-stacking constraint — but we are approaching the
point where further refinement of a *free, reversible, non-fix* mitigation is
worth less than the decision it is standing in for. Today's episodes were
55–120 minutes. Nothing in this section touches those.

Position unchanged and final from me unless new evidence arrives:

1. Canonicalisation — done, verified.
2. Check `interval 30s` / `timeout 25s` / `grace_period 60s`, gunicorn untouched
   — ready to apply, awaiting the owner.
3. **Second web machine** — the only item addressing the multi-hour episodes.
4. Instrumentation, then revisit gunicorn's timeout with data; re-tighten checks
   once failover exists.
