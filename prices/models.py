from django.db import models


# Create your models here.
class Forecast(models.Model):
    forecast_date = models.DateField()

    def __str__(self) -> str:
        return super().__str__()
