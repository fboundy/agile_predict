from django.db import models

# class Forecast(models.Model):
#     name = models.CharField(unique=True, max_length=64)
#     # source = models.CharField(max_length=32)
#     # source = models.CharField(max_length=32, choices=DataSource.choices)


class Forecasts(models.Model):
    name = models.CharField(unique=True, max_length=64)
    # forecast = models.ForeignKey(Forecast, on_delete=CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    # expired = models.BooleanField(default=False)


# class TSValue(models.Model):
#     version = models.ForeignKey(TSVersion, on_delete=CASCADE)
#     time = models.DateTimeField()
#     value = models.FloatField()


class PriceHistory(models.Model):
    date_time = models.DateTimeField(unique=True)
    day_ahead = models.FloatField()
    agile = models.FloatField()


class History(models.Model):
    date_time = models.DateTimeField(unique=True)
    wind = models.FloatField()
    bm_wind = models.FloatField()
    solar = models.FloatField()
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()
    demand_source = models.CharField(max_length=10)


class ForecastData(models.Model):
    forecast = models.ForeignKey(Forecasts, on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    day_ahead = models.FloatField()
    agile = models.FloatField()
    wind = models.FloatField()
    bm_wind = models.FloatField()
    solar = models.FloatField()
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()
