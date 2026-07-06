from django.http import HttpResponse


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
