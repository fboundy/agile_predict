from django.db import models
from django.urls import reverse


class UpdateJob(models.Model):
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
        return f"{self.id}: {self.status}"


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
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()
    # other_gen_capacity = models.FloatField()
    # intercon_capacity = models.FloatField()

    # def __str__(self):
    #     return f"{self.date_time.strftime('%Y-%m-%dT%H:%M%Z') {self.agile:5.2f}}"
