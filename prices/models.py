from django.db import models
from django.urls import reverse


class UpdateJob(models.Model):
    JOB_UPDATE = "update"
    JOB_LATEST_AGILE = "latest_agile"

    JOB_TYPE_CHOICES = [
        (JOB_UPDATE, "Full update"),
        (JOB_LATEST_AGILE, "Latest Agile prices"),
    ]

    STATUS_PENDING = "pending"
    STATUS_RUNNING = "running"
    STATUS_COMPLETED = "completed"
    STATUS_FAILED = "failed"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_RUNNING, "Running"),
        (STATUS_COMPLETED, "Completed"),
        (STATUS_FAILED, "Failed"),
    ]

    job_type = models.CharField(max_length=32, choices=JOB_TYPE_CHOICES, default=JOB_UPDATE)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    options = models.JSONField(default=dict, blank=True)
    error = models.TextField(blank=True)
    log_file = models.CharField(max_length=255, blank=True)
    requested_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-requested_at"]

    def __str__(self):
        return f"{self.id}: {self.job_type} {self.status}"


class PlotImage(models.Model):
    filename = models.CharField(max_length=255, unique=True)
    content_type = models.CharField(max_length=64, default="image/png")
    content = models.BinaryField()
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["filename"]

    def __str__(self):
        return self.filename


class ExternalForecast(models.Model):
    SOURCE_AGILEFORECAST = "agileforecast"
    SOURCE_X2R = "x2r"

    SOURCE_CHOICES = [
        (SOURCE_AGILEFORECAST, "AgileForecast"),
        (SOURCE_X2R, "X2R"),
    ]

    source = models.CharField(max_length=32, choices=SOURCE_CHOICES)
    region = models.CharField(max_length=2)
    forecast_name = models.CharField(max_length=128, blank=True)
    source_created_at = models.DateTimeField()
    downloaded_at = models.DateTimeField(auto_now_add=True)
    date_time = models.DateTimeField()
    agile_pred = models.FloatField()
    agile_low = models.FloatField(null=True, blank=True)
    agile_high = models.FloatField(null=True, blank=True)

    class Meta:
        indexes = [
            models.Index(fields=["source", "region", "source_created_at"]),
            models.Index(fields=["source", "region", "date_time"]),
        ]
        unique_together = ("source", "region", "source_created_at", "date_time")


class RequestMetric(models.Model):
    SURFACE_WEB = "web"
    SURFACE_API = "api"
    SURFACE_UPDATE = "update"
    SURFACE_ADMIN = "admin"
    SURFACE_STATIC = "static"

    SURFACE_CHOICES = [
        (SURFACE_WEB, "Web"),
        (SURFACE_API, "API"),
        (SURFACE_UPDATE, "Update"),
        (SURFACE_ADMIN, "Admin"),
        (SURFACE_STATIC, "Static"),
    ]

    date = models.DateField()
    hour = models.PositiveSmallIntegerField()
    surface = models.CharField(max_length=16, choices=SURFACE_CHOICES)
    path = models.CharField(max_length=255)
    method = models.CharField(max_length=8)
    status_code = models.PositiveSmallIntegerField()
    request_count = models.PositiveIntegerField(default=0)

    class Meta:
        indexes = [
            models.Index(fields=["date", "surface"]),
            models.Index(fields=["date", "hour"]),
        ]
        unique_together = ("date", "hour", "surface", "path", "method", "status_code")
        ordering = ["-date", "-hour", "surface", "path"]

    def __str__(self):
        return f"{self.date} {self.hour:02d}:00 {self.surface} {self.path} {self.request_count}"


class RequestClientSeen(models.Model):
    date = models.DateField()
    surface = models.CharField(max_length=16, choices=RequestMetric.SURFACE_CHOICES)
    client_hash = models.CharField(max_length=64)

    class Meta:
        indexes = [
            models.Index(fields=["date", "surface"]),
        ]
        unique_together = ("date", "surface", "client_hash")
        ordering = ["-date", "surface"]


class Forecasts(models.Model):
    name = models.CharField(unique=True, max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    mean = models.FloatField(null=True)
    stdev = models.FloatField(null=True)

    def __str__(self):
        return self.name


class PriceHistory(models.Model):
    date_time = models.DateTimeField(unique=True)
    day_ahead = models.FloatField()
    agile = models.FloatField()


class AgileData(models.Model):
    forecast = models.ForeignKey(Forecasts, related_name="prices", on_delete=models.CASCADE)
    region = models.CharField(max_length=1)
    agile_pred = models.FloatField()
    agile_low = models.FloatField()
    agile_high = models.FloatField()
    date_time = models.DateTimeField()

    def get_absolute_url(self):
        return reverse("graph", kwargs={"region": self.region})


class History(models.Model):
    date_time = models.DateTimeField(unique=True)
    total_wind = models.FloatField()
    bm_wind = models.FloatField()
    solar = models.FloatField()
    nuclear = models.FloatField(default=0)
    gas_ttf = models.FloatField(null=True, blank=True)
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()
    # other_gen_capacity = models.FloatField()
    # intercon_capacity = models.FloatField()


class ForecastData(models.Model):
    forecast = models.ForeignKey(Forecasts, related_name="data", on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    day_ahead = models.FloatField(null=True)
    bm_wind = models.FloatField()
    solar = models.FloatField()
    emb_wind = models.FloatField()
    nuclear = models.FloatField(default=0)
    gas_ttf = models.FloatField(null=True, blank=True)
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()
    # other_gen_capacity = models.FloatField()
    # intercon_capacity = models.FloatField()

    # def __str__(self):
    #     return f"{self.date_time.strftime('%Y-%m-%dT%H:%M%Z') {self.agile:5.2f}}"
