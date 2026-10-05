from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *


class Command(BaseCommand):
    # def add_arguments(self, parser):
    # Positional arguments
    # parser.add_argument("poll_ids", nargs="+", type=int)

    # Named (optional) arguments
    # parser.add_argument(
    #     "--delete",
    #     action="store_true",
    #     help="Only keep one forecast per day ",
    # )

    def handle(self, *args, **options):
        df = queryset_to_df(ForecastData.objects.all())[["forecast_id", "day_ahead"]]
        # print(df.columns)
        #
        df["agile_pred"] = day_ahead_to_agile(df["day_ahead"], region="X")
        df["region"] = "X"
        for id in df["forecast_id"].drop_duplicates():
            x = df[df["forecast_id"] == id].copy()
            x["forecast"] = Forecasts.objects.filter(id=id)[0]
            x = x.drop(["forecast_id", "day_ahead"], axis=1)
            print(x)
            df_to_Model(x, AgileData)
