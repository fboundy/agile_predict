from django.contrib import admin

# Register your models here.
from .models import Forecasts, History, PriceHistory, ForecastData

admin.site.register(Forecasts)
admin.site.register(ForecastData)
