from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *
import os


class Command(BaseCommand):
    def handle(self, *args, **options):
        local_dir = os.path.join(os.getcwd(), ".local")
        hdf = os.path.join(local_dir, "forecast.hdf")
        if not os.path.exists(local_dir):
            os.mkdir(".local")

        elif os.path.exists(hdf):
            os.remove(hdf)

        for data in [AgileData, Forecasts, ForecastData, PriceHistory]:
            key = data.__name__
            print(key)

            if key == "AgileData":
                ff = list(pd.DataFrame(list(Forecasts.objects.all().values())).sort_values("created_at")["id"][-10:])
                print(ff)
                df = pd.DataFrame(list(data.objects.filter(forecast_id__in=ff).values()))
            else:
                df = pd.DataFrame(list(data.objects.all().values()))

            try:
                df.to_hdf(hdf, key=key, mode="a")
                print(f"{key:20s}: {len(df):>10d} OK")
            except:
                print(f"{key:20s}: {len(df):>10d} <<<<< Error!")
