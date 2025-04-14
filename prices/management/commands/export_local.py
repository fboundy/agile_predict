from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *
import os


class Command(BaseCommand):
    def handle(self, *args, **options):
        local_dir = os.path.join(os.getcwd(), "temp")
        hdf = os.path.join(local_dir, "forecast.hdf")
        if not os.path.exists(local_dir):
            os.mkdir("temp")

        elif os.path.exists(hdf):
            os.remove(hdf)

        for data in [AgileData, Forecasts, ForecastData, PriceHistory]:
            key = data.__name__
            print(key)

            df = pd.DataFrame(list(data.objects.all().values()))
            if key == "AgileData":
                df = df[df["region"].isin(["G", "X"])]

            try:
                df.to_hdf(hdf, key=key, mode="a")
                print(f"{key:20s}: {len(df):>10d} OK")
            except:
                print(f"{key:20s}: {len(df):>10d} <<<<< Error!")
