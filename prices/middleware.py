import hashlib
import hmac

from django.conf import settings
from django.db import IntegrityError, OperationalError, ProgrammingError, transaction
from django.db.models import F
from django.utils import timezone

from .models import RequestClientSeen, RequestMetric


IGNORED_PREFIXES = (
    "/static/",
    "/favicon.ico",
)


def classify_surface(path):
    if path.startswith("/api/"):
        return RequestMetric.SURFACE_API
    if path.startswith("/update"):
        return RequestMetric.SURFACE_UPDATE
    if path.startswith("/admin/"):
        return RequestMetric.SURFACE_ADMIN
    if path.startswith("/static/"):
        return RequestMetric.SURFACE_STATIC
    return RequestMetric.SURFACE_WEB


def normalize_path(path):
    if path.startswith("/api/"):
        parts = path.strip("/").split("/")
        return "/" + "/".join(parts[:2]) + "/"
    if path.startswith("/stats/plot/"):
        return "/stats/plot/"
    if len(path) > 255:
        return path[:255]
    return path


def client_identifier(request):
    forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR", "")
    ip_address = forwarded_for.split(",", 1)[0].strip() or request.META.get("REMOTE_ADDR", "")
    user_agent = request.META.get("HTTP_USER_AGENT", "")
    return f"{ip_address}|{user_agent}"


def hash_client(date, request):
    key = settings.SECRET_KEY.encode("utf-8")
    message = f"{date.isoformat()}|{client_identifier(request)}".encode("utf-8", errors="ignore")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


class RequestMetricsMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        self.record_request(request, response)
        return response

    def record_request(self, request, response):
        path = request.path_info or "/"
        if path.startswith(IGNORED_PREFIXES):
            return

        now = timezone.localtime()
        date = now.date()
        surface = classify_surface(path)
        path = normalize_path(path)
        status_code = getattr(response, "status_code", 0) or 0

        try:
            with transaction.atomic():
                metric, created = RequestMetric.objects.get_or_create(
                    date=date,
                    hour=now.hour,
                    surface=surface,
                    path=path,
                    method=request.method,
                    status_code=status_code,
                    defaults={"request_count": 1},
                )
                if not created:
                    RequestMetric.objects.filter(pk=metric.pk).update(request_count=F("request_count") + 1)

                RequestClientSeen.objects.get_or_create(
                    date=date,
                    surface=surface,
                    client_hash=hash_client(date, request),
                )
        except (IntegrityError, OperationalError, ProgrammingError):
            return
