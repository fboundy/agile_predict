from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *


class Command(BaseCommand):
    def handle(self, *args, **options):
        dfs = []
        for d in pd.date_range("2024-10-01", "2024-10-14"):
            data = {
                "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/FOU2T14D?format=json",
                "params": {
                    "publishDate": pd.Timestamp(d).strftime("%Y-%m-%d"),
                    "fuelType": "NUCLEAR",
                },
                "record_path": ["data"],
                "cols": ["forecastDate", "fuelType", "outputUsable", "publishTime"],
                "date_col": "forecastDate",
                "resample": "30min",
                "func": "ffill",
                "tz": "UTC",
            }

            df, _ = DataSet(**data).download()
            # dfs += [DataSet(**data).download()]

            # print
            # df = pd.concat(dfs)
            print(df)
        # for f in Forecasts.objects.all():
        #     print(f.name)
        #     d = ForecastData.objects.filter(forecast=f).order_by("date_time")
        #     print(d[0].date_time)
