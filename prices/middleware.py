import logging
import time

from django.conf import settings
from django.core.cache import cache, caches
from django.http import HttpResponse
from django.utils.cache import patch_response_headers
from django.utils.dateparse import parse_date

from . import blocklist

logger = logging.getLogger("prices.ratelimit")


class HealthCheckMiddleware:
    """Answer /healthz before any other middleware runs.

    Fly.io's health-check prober hits this path using the machine's private
    network address as the Host header, which Django's ALLOWED_HOSTS check
    (triggered by CommonMiddleware/SecurityMiddleware) rejects with
    DisallowedHost. That made the health check permanently "critical", so
    Fly's proxy stopped routing any real traffic to the machine. Short-circuit
    here, first in MIDDLEWARE, so the request never reaches host validation.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path == "/healthz":
            return HttpResponse("OK")
        return self.get_response(request)


def client_ip(request):
    """Best-effort real client IP behind the fly.io proxy."""
    ip = request.META.get("HTTP_FLY_CLIENT_IP")
    if ip:
        return ip.strip()
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR", "") or "unknown"


class BlocklistMiddleware:
    """Reject requests from known-malicious IPs (reputation blocklist).

    Complements the rate limiter: the limiter stops volumetric abuse, this drops
    traffic from IP ranges on a curated blocklist (default FireHOL level 1)
    outright. The check is a cheap in-memory bisect (see prices/blocklist.py);
    the list refreshes in a background thread. Fail-open; `BLOCKLIST_ENFORCE=false`
    logs would-be blocks without dropping them.
    """

    EXEMPT_PREFIXES = ("/healthz", "/static/", "/favicon")

    def __init__(self, get_response):
        self.get_response = get_response
        self.enabled = getattr(settings, "BLOCKLIST_ENABLED", False)
        self.enforce = getattr(settings, "BLOCKLIST_ENFORCE", True)

    def __call__(self, request):
        if not self.enabled or request.path.startswith(self.EXEMPT_PREFIXES):
            return self.get_response(request)
        ip = client_ip(request)
        try:
            blocked = blocklist.is_blocked(ip)
        except Exception:  # pragma: no cover - must fail open
            blocked = False
        if blocked:
            logger.info(
                "BLOCKLIST-DENY ip=%s enforce=%s path=%s ua=%r",
                ip, self.enforce, request.path,
                request.META.get("HTTP_USER_AGENT", "")[:200],
            )
            if self.enforce:
                return HttpResponse("Forbidden\n", status=403, content_type="text/plain")
        return self.get_response(request)


class RateLimitMiddleware:
    """Per-IP fixed-window rate limiter to protect the 2 gunicorn workers.

    A crawler sweeping every region x filter combination can occupy both sync
    workers with expensive chart renders, tripping gunicorn's WORKER TIMEOUT and
    taking the whole machine offline (fly proxy then reports "no healthy
    instances"). This caps how many throttled requests a single IP may make per
    minute, tracks repeat offenders, and escalates to a longer block for the
    worst ones.

    Design notes:
    - Fail-open: any cache/backend error lets the request through — throttling
      must never itself take the site down.
    - Requests carrying a session cookie (logged-in admins) are exempt.
    - Cheap/static paths are exempt; only GET/HEAD are counted.
    - Counters live in the shared file cache so both workers share state.
    """

    EXEMPT_PREFIXES = (
        "/healthz",
        "/static/",
        "/favicon",
        "/robots",
        "/apple-touch",
        "/.well-known",
        "/accounts/",
        "/update",
    )

    def __init__(self, get_response):
        self.get_response = get_response
        self.enabled = getattr(settings, "RATELIMIT_ENABLED", True)
        self.enforce = getattr(settings, "RATELIMIT_ENFORCE", True)
        self.limit = getattr(settings, "RATELIMIT_PER_MIN", 60)
        self.block_threshold = getattr(settings, "RATELIMIT_BLOCK_THRESHOLD", 5)
        self.block_seconds = getattr(settings, "RATELIMIT_BLOCK_SECONDS", 600)

    def __call__(self, request):
        if not self.enabled or request.method not in ("GET", "HEAD"):
            return self.get_response(request)
        if request.path.startswith(self.EXEMPT_PREFIXES) or "sessionid" in request.COOKIES:
            return self.get_response(request)

        ip = client_ip(request)
        try:
            decision = self._check(ip, request)
        except Exception:  # pragma: no cover - defensive, must fail open
            logger.exception("rate-limit check errored; failing open")
            return self.get_response(request)

        if decision is not None:
            return decision
        return self.get_response(request)

    def _incr(self, key, timeout):
        """Increment a counter, creating it if absent. Not strictly atomic
        across workers on the file cache, but good enough for throttling."""
        if cache.add(key, 1, timeout=timeout):
            return 1
        try:
            return cache.incr(key)
        except ValueError:
            cache.set(key, 1, timeout=timeout)
            return 1

    def _check(self, ip, request):
        if cache.get(f"rl:block:{ip}"):
            return self._reject(ip, request, reason="blocked")

        window = int(time.time() // 60)
        count = self._incr(f"rl:cnt:{ip}:{window}", timeout=120)
        if count <= self.limit:
            return None

        # Over the per-minute limit — record the offence and maybe escalate.
        offences = self._incr(f"rl:off:{ip}", timeout=3600)
        if offences >= self.block_threshold and not cache.get(f"rl:block:{ip}"):
            cache.set(f"rl:block:{ip}", 1, timeout=self.block_seconds)
            logger.warning(
                "BLOCK ip=%s offences=%s for=%ss path=%s ua=%r",
                ip, offences, self.block_seconds, request.path,
                request.META.get("HTTP_USER_AGENT", "")[:200],
            )
        else:
            logger.info(
                "OVER-LIMIT ip=%s count=%s/%s offences=%s path=%s ua=%r",
                ip, count, self.limit, offences, request.path,
                request.META.get("HTTP_USER_AGENT", "")[:200],
            )
        return self._reject(ip, request, reason="rate")

    def _reject(self, ip, request, reason):
        if not self.enforce:
            return None  # log-only mode
        resp = HttpResponse(
            "Too many requests — please slow down.\n",
            status=429,
            content_type="text/plain",
        )
        resp["Retry-After"] = "60"
        return resp


_TRUE_WORDS = {"1", "true", "yes", "on"}

# Mirrors HistoryView.window_options keys.
_HISTORY_WINDOWS = ("last-week", "last-2-weeks", "last-month", "custom")


def _canonical_query(path, q):
    """Collapse query strings that produce identical output onto one cache key.

    The raw query string made `?days=5&gen=1`, `?gen=1&days=5` and `?days=5`
    (with gen defaulting to on) three separate entries for byte-identical pages,
    so any client permuting parameters — or spelling out defaults — defeated the
    cache. That is exactly the traffic pattern seen during the outages.

    Each parameter is normalised with the SAME semantics as the view that reads
    it, which differ between surfaces (e.g. `export` is only true for a literal
    "1" on the v2 chart, but accepts true/yes/on in the API). Getting that wrong
    would serve the wrong cached page, so every parameter below mirrors its view
    exactly and is covered by tests. Unknown parameters are dropped because the
    views never read them. Returns None when the surface is unknown or a value
    cannot be parsed, in which case the caller falls back to the raw query
    string rather than risk collapsing two different pages together.
    """
    try:
        if path.startswith("/api/"):
            # api/views.py: PriceForecast*APIView / AccuracyAPIView
            days = min(max(int(q.get("days", 14)), 1), 14)
            count = int(q.get("forecast_count", 1))
            high_low = "1" if str(q.get("high_low", "true")).lower() in {"true", "1"} else "0"
            export = "1" if str(q.get("export", "false")).lower() in _TRUE_WORDS else "0"
            region = str(q.get("region", "X")).upper()
            parts = [
                f"days={days}", f"forecast_count={count}",
                f"high_low={high_low}", f"export={export}", f"region={region}",
            ]
            fmt = q.get("format", "")
            if fmt:
                parts.append(f"format={fmt}")  # DRF renderer — changes output
            return "&".join(parts)

        if path.startswith("/v2/history/"):
            # prices/views.py: HistoryView.get_date_window / get_context_data,
            # plus HistoryV2View.get_context_data for metric + unit_mode.
            #
            # MUST stay ahead of the /v2/ branch below. The history page shares
            # the /v2/ prefix but takes a completely different parameter set, so
            # it used to fall through to the chart branch, which dropped every
            # one of these as "unknown". Each history URL then canonicalised to
            # the same chart-default string and they all shared ONE cache entry:
            # whichever variant rendered first was served for every window,
            # lead time, metric and unit toggle until the key rolled.
            window = q.get("window", "last-2-weeks")
            if window not in _HISTORY_WINDOWS:
                window = "last-2-weeks"

            date_parts = []
            if window == "custom":
                # get_date_window only honours custom dates when BOTH parse and
                # are correctly ordered, and otherwise falls back to the
                # 2-week window — so anything else must collapse onto that key.
                start_date = parse_date(q.get("start_date", ""))
                end_date = parse_date(q.get("end_date", ""))
                if start_date and end_date and start_date <= end_date:
                    date_parts = [
                        f"start_date={start_date.isoformat()}",
                        f"end_date={end_date.isoformat()}",
                    ]
                else:
                    window = "last-2-weeks"

            # HistoryView.max_offset_days is 14; mirrored here like the other
            # view constants in this function.
            offset_days = min(max(int(q.get("offset_days", 1)), 0), 14)
            metric = str(q.get("metric", "mae")).lower()
            if metric not in ("mae", "rmse"):
                metric = "mae"
            unit_mode = str(q.get("unit_mode", "da")).lower()
            if unit_mode not in ("agile", "da"):
                unit_mode = "da"
            cmp_af = "1" if str(q.get("compare_agileforecast")).lower() in _TRUE_WORDS else "0"
            cmp_x2r = "1" if str(q.get("compare_x2r")).lower() in _TRUE_WORDS else "0"
            # DEV ONLY overlay (prices/postprocess.py) — see the /v2/ branch's `dac`
            # comment below; harmless to include on a build that lacks it.
            cmp_corr = "1" if str(q.get("compare_corrected")).lower() in _TRUE_WORDS else "0"

            return "&".join(
                [f"window={window}", *date_parts, f"offset_days={offset_days}",
                 f"metric={metric}", f"unit_mode={unit_mode}",
                 f"compare_agileforecast={cmp_af}", f"compare_x2r={cmp_x2r}",
                 f"compare_corrected={cmp_corr}"]
            )

        if path.startswith("/v2/"):
            # prices/views.py: GraphV2View.get_context_data / ext_forecast_json
            # Catch-all for the remaining /v2/ surfaces. Safe only because every
            # other v2 page (stats, about, model, api_how_to, limitations)
            # ignores the query string entirely. A NEW v2 page that reads its own
            # parameters needs its own branch above, or its URLs will collapse
            # together exactly as the history page's did.
            days = min(max(int(q.get("days", 5)), 1), 14)
            band = "0" if q.get("band", "1") == "0" else "1"
            export = "1" if q.get("export", "0") == "1" else "0"
            gen = "1" if q.get("gen", "1") == "1" else "0"
            fg = "1" if q.get("fg", "0") == "1" else "0"
            dc = "1" if q.get("dc", "0") == "1" else "0"
            af = "1" if str(q.get("af", "")).lower() in _TRUE_WORDS else "0"
            x2r = "1" if str(q.get("x2r", "")).lower() in _TRUE_WORDS else "0"
            overlap = "1" if q.get("overlap", "0") == "1" else "0"
            # Post-processed day-ahead overlay (prices/postprocess.py). DEV ONLY: the
            # underlying column and view code only exist on the dev branch, but the
            # param is harmless to include everywhere — it is simply always "0" on a
            # build that lacks it, since the view never sets show_corrected=True there.
            dac = "1" if q.get("dac", "0") == "1" else "0"
            # Summary-card placement (GH #93). It changes the rendered HTML, so
            # it has to be in the key or all three positions collapse onto one
            # cached entry. Mirrors GraphV2View's validation, including the
            # fallback, so an unrecognised value shares the default's entry
            # rather than minting a new one per junk string.
            summary = q.get("summary", "above")
            if summary not in ("above", "below", "off"):
                summary = "above"
            # Forecast ids are used as an `id__in` filter, so order is irrelevant.
            fc = sorted({int(x) for x in q.getlist("fc") if str(x).strip()})
            parts = [
                f"days={days}", f"band={band}", f"export={export}", f"gen={gen}",
                f"fg={fg}", f"dc={dc}", f"af={af}", f"x2r={x2r}", f"overlap={overlap}",
                f"summary={summary}", f"dac={dac}",
            ]
            if fc:
                parts.append("fc=" + ",".join(str(i) for i in fc))
            return "&".join(parts)
    except (TypeError, ValueError):
        return None
    return None


class ResponseCacheMiddleware:
    """Event-keyed cache of anonymous GET responses for the heavy chart/API paths.

    The v2 chart pages and JSON API build Plotly figures + run sizeable queries
    on every hit (~1.2s of CPU on the shared vCPU), but their content only
    actually changes when (a) a new forecast / fresh Agile prices land (a
    handful of times a day) or (b) the half-hour slot rolls over (the "Now"
    marker, current-price card, upcoming-window calcs). So the cache key embeds
    a data-version stamp and the current 30-min slot: entries are reused until
    the content genuinely changes — a far better hit-rate than any fixed TTL,
    with zero staleness right after an update lands. The stored TTL is only a
    safety net; browsers get a separate short client TTL.

    Only cached when safe: anonymous (no session), GET, 200, no Set-Cookie
    (avoids freezing a CSRF/session cookie), body under a size cap, and on an
    allow-listed prefix. The health page is explicitly excluded so it stays live.
    """

    CACHEABLE_PREFIXES = ("/v2/", "/api/")
    # Ops pages show live cross-server / API data — never serve them stale.
    EXCLUDE_PREFIXES = ("/v2/health", "/v2/costs")

    def __init__(self, get_response):
        self.get_response = get_response
        self.enabled = getattr(settings, "RESPONSE_CACHE_ENABLED", True)
        self.ttl = getattr(settings, "RESPONSE_CACHE_TTL", 3600)
        self.client_ttl = getattr(settings, "RESPONSE_CACHE_CLIENT_TTL", 300)
        self.max_bytes = getattr(settings, "RESPONSE_CACHE_MAX_BYTES", 2_000_000)

    def _cacheable_request(self, request):
        if not self.enabled or request.method != "GET":
            return False
        if "sessionid" in request.COOKIES or request.user.is_authenticated:
            return False
        path = request.path
        if path.startswith(self.EXCLUDE_PREFIXES):
            return False
        return path.startswith(self.CACHEABLE_PREFIXES)

    def _store(self):
        """Shared across workers, so a URL is rendered once for the machine
        rather than once per worker."""
        try:
            return caches["responses"]
        except Exception:  # pragma: no cover - fall back to the default cache
            return cache

    def _data_version(self):
        """Cheap stamp of the underlying data: bumps when a new forecast or
        fresh Agile prices land. Micro-cached for 30s so the per-request cost is
        ~zero (worst case a new forecast shows ≤30s late). Fail-open to a static
        stamp — the slot bucket in the key still bounds staleness to 30 min."""
        try:
            ver = cache.get("rc:data-ver")
            if ver is None:
                from prices.models import Forecasts, PriceHistory

                fid = Forecasts.objects.order_by("-created_at").values_list("id", flat=True).first() or 0
                pts = PriceHistory.objects.order_by("-date_time").values_list("date_time", flat=True).first()
                ver = f"{fid}-{int(pts.timestamp()) if pts else 0}"
                cache.set("rc:data-ver", ver, 30)
            return ver
        except Exception:  # pragma: no cover - fail open
            return "na"

    def _key(self, request):
        # Half-hour bucket: the only *time*-dependence the pages have (the "Now"
        # marker and current-slot pricing) changes exactly on the half hour.
        slot = int(time.time() // 1800)
        canon = _canonical_query(request.path, request.GET)
        qs = canon if canon is not None else request.META.get("QUERY_STRING", "")
        return (
            f"rc:{request.get_host()}:{request.path}?{qs}"
            f":v{self._data_version()}:s{slot}"
        )

    def __call__(self, request):
        if not self._cacheable_request(request):
            return self.get_response(request)

        key = self._key(request)
        store = self._store()
        try:
            cached = store.get(key)
        except Exception:  # pragma: no cover - fail open
            cached = None
        if cached is not None:
            body, status, content_type = cached
            resp = HttpResponse(body, status=status, content_type=content_type)
            resp["X-Response-Cache"] = "HIT"
            patch_response_headers(resp, cache_timeout=self.client_ttl)
            return resp

        response = self.get_response(request)
        try:
            if (
                response.status_code == 200
                and not response.cookies
                and not response.streaming
                and not response.has_header("Set-Cookie")
                and len(response.content) <= self.max_bytes
            ):
                store.set(
                    key,
                    (response.content, response.status_code, response.get("Content-Type", "text/html")),
                    timeout=self.ttl,
                )
                # Browsers get a short client TTL — they can't see the version
                # key, so a long client cache would delay new forecasts.
                patch_response_headers(response, cache_timeout=self.client_ttl)
                response["X-Response-Cache"] = "MISS"
        except Exception:  # pragma: no cover - never fail because of caching
            logger.exception("response cache store failed")
        return response
