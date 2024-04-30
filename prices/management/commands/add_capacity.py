from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *


class Command(BaseCommand):
    # def add_arguments(self, parser):
    #     # Positional arguments
    #     # parser.add_argument("poll_ids", nargs="+", type=int)

    #     # Named (optional) arguments
    #     parser.add_argument(
    #         "--delete",
    #         action="store_true",
    #         help="Only keep one forecast per day ",
    #     )

    def handle(self, *args, **options):
        for h in History.objects.all()[:1]:
            print(h.date_time)
            data = {
                "url": "https://data.elexon.co.uk/bmrs/api/v1/forecast/availability/daily/history",
                "params": {
                    "publishTime": (h.date_time - pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
                    "level": "fuelType",
                },
                "record_path": ["data"],
                "cols": "ND",
            }

            cap = DataSet(**data).download()
            cap = cap[cap["forecastDate"] == h.date_time.strftime("%Y-%m-%d")].set_index("fuelType")["outputUsable"]

            intercon_capacity = cap[cap.index.str.contains("^INT")].sum()
            other_gen_capacity = cap.sum() - intercon_capacity - cap.loc["WIND"]
            h.intercon_capacity = intercon_capacity
            h.other_gen_capacity = other_gen_capacity
            h.save()

        for f in Forecasts.objects.all()[:1]:
            print(f.name)
            d = ForecastData.objects.filter(forecast=f).order_by("date_time")
            print(d[0].date_time)
