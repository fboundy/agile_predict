"""Reputation-based IP blocklist.

Loads a conservative, curated list of known-malicious IPv4 ranges (default:
FireHOL level 1) and offers an O(log n) membership test for the request path.

Design goals, given this runs on a worker-constrained fly.io app:
- The request path only does an in-memory bisect — never a network call.
- Refreshing happens in a background daemon thread on a TTL, so a slow or failed
  download can never block or break requests (fail-open throughout).
- A pickled copy is cached on disk (shared by the machine's workers) so a freshly
  started/recycled worker is protected immediately from the last-known list.
- IPv4 only; IPv6 clients simply never match (fail-open) to keep it simple.
"""

import bisect
import ipaddress
import logging
import os
import pickle
import threading
import time
import urllib.request

from django.conf import settings

logger = logging.getLogger("prices.ratelimit")

# Module-level state, swapped atomically under _lock.
_state = {
    "ranges": [],   # sorted, merged list of (start_int, end_int)
    "starts": [],   # start_int values, for bisect
    "count": 0,
    "loaded_at": 0.0,
    "refreshing": False,
}
_lock = threading.Lock()


def _cache_file():
    cache_dir = getattr(settings, "CACHE_DIR", "/tmp")
    return os.path.join(cache_dir, "blocklist.pickle")


def _merge(ranges):
    """Sort and coalesce adjacent/overlapping (start, end) integer ranges."""
    ranges.sort()
    merged = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _parse(text):
    ranges = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] in "#;":
            continue
        token = line.split()[0].split(";")[0].strip()
        try:
            net = ipaddress.ip_network(token, strict=False)
        except ValueError:
            continue
        if net.version != 4:
            continue
        ranges.append((int(net.network_address), int(net.broadcast_address)))
    return ranges


def _apply(merged):
    with _lock:
        _state["ranges"] = merged
        _state["starts"] = [s for s, _ in merged]
        _state["count"] = len(merged)
        _state["loaded_at"] = time.time()


def _load_cache():
    try:
        with open(_cache_file(), "rb") as fh:
            merged = pickle.load(fh)
    except Exception:
        return False
    if isinstance(merged, list):
        _apply(merged)
        return True
    return False


def _download():
    urls = getattr(settings, "BLOCKLIST_URLS", [])
    collected = []
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "agilepredict-blocklist/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                text = resp.read().decode("utf-8", "replace")
            collected.extend(_parse(text))
        except Exception as exc:  # network/parse errors must not break anything
            logger.warning("blocklist fetch failed url=%s err=%s", url, exc)
    return _merge(collected) if collected else None


def _refresh():
    try:
        merged = _download()
        if merged:
            _apply(merged)
            try:
                with open(_cache_file(), "wb") as fh:
                    pickle.dump(merged, fh)
            except OSError:
                pass
            logger.info("blocklist refreshed entries=%s", len(merged))
    finally:
        with _lock:
            _state["refreshing"] = False


def _maybe_refresh():
    ttl = getattr(settings, "BLOCKLIST_TTL", 21600)
    with _lock:
        if _state["refreshing"]:
            return
        fresh = _state["count"] and (time.time() - _state["loaded_at"] < ttl)
        if fresh:
            return
        _state["refreshing"] = True
    # Nothing loaded yet? Warm-start from the on-disk cache before the network call.
    if not _state["count"]:
        _load_cache()
    threading.Thread(target=_refresh, daemon=True).start()


def refresh_now():
    """Synchronous refresh — for the management command / manual use."""
    _refresh()
    return _state["count"]


def is_blocked(ip):
    """True if `ip` is in the blocklist. Fail-open on any error."""
    if not getattr(settings, "BLOCKLIST_ENABLED", False):
        return False
    try:
        addr = ipaddress.ip_address(ip)
        # Never block private/loopback/reserved/link-local addresses. Curated
        # lists include these bogon ranges, but seeing one as a client IP means
        # local/LAN traffic (or a proxy fallback), which must never be blocked.
        if not addr.is_global:
            return False
        _maybe_refresh()
        starts = _state["starts"]
        ranges = _state["ranges"]
        if not starts:
            return False
        ip_int = int(addr)
    except Exception:
        return False
    idx = bisect.bisect_right(starts, ip_int) - 1
    return idx >= 0 and ranges[idx][0] <= ip_int <= ranges[idx][1]


def status():
    return {"count": _state["count"], "loaded_at": _state["loaded_at"]}
