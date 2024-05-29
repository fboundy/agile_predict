from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *
import os


class Command(BaseCommand):
    def handle(self, *args, **options):
        print(os.getcwd())

        hdf = os.path.join(os.getcwd(), ".local", "forecast.hdf")

        for data in [Forecasts, ForecastData, AgileData, PriceHistory]:
            df = pd.DataFrame(list(data.objects.all().values()))
            df.to_hdf(hdf, key=data.__name__, mode="a")
