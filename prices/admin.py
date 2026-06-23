from django.contrib import admin

# Register your models here.
from .models import Forecasts, ForecastData, History, PriceHistory, RequestClientSeen, RequestMetric

admin.site.register(Forecasts)
admin.site.register(ForecastData)
admin.site.register(History)
admin.site.register(PriceHistory)
admin.site.register(RequestMetric)
admin.site.register(RequestClientSeen)
