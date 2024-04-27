from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *


class Command(BaseCommand):
    def handle(self, *args, **options):
        for f in Forecasts.objects.all():
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            print(f.name, q.count(), a.count())
            if q.count() < 600 or a.count() < 8000:
                f.delete()
