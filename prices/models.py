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
    def __str__(self):
        return self.name


# class TSValue(models.Model):
#     version = models.ForeignKey(TSVersion, on_delete=CASCADE)
#     time = models.DateTimeField()
#     value = models.FloatField()


class PriceHistory(models.Model):
    date_time = models.DateTimeField(unique=True)
    day_ahead = models.FloatField()
    agile = models.FloatField()


class AgileData(models.Model):
    forecast = models.ForeignKey(Forecasts, related_name="data", on_delete=models.CASCADE)
    region = models.CharField(_(""), max_length=1)
    agile_pred = models.FloatField()


class History(models.Model):
    date_time = models.DateTimeField(unique=True)
    # wind = models.FloatField()
    bm_wind = models.FloatField()
    solar = models.FloatField()
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()


class ForecastData(models.Model):
    forecast = models.ForeignKey(Forecasts, related_name="data", on_delete=models.CASCADE)
    date_time = models.DateTimeField()
    agile_pred = models.FloatField()
    agile_actual = models.FloatField(null=True)
    # wind = models.FloatField()
    bm_wind = models.FloatField()
    solar = models.FloatField()
    temp_2m = models.FloatField()
    wind_10m = models.FloatField()
    rad = models.FloatField()
    demand = models.FloatField()

    # def __str__(self):
    #     return f"{self.date_time.strftime('%Y-%m-%dT%H:%M%Z') {self.agile:5.2f}}"
