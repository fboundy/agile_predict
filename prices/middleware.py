import logging
import time

from django.conf import settings
from django.core.cache import cache, caches
from django.http import HttpResponse
from django.utils.cache import patch_response_headers

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
        return (
            f"rc:{request.get_host()}:{request.path}?{request.META.get('QUERY_STRING', '')}"
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
