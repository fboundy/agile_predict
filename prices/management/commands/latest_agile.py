from django.core.management.base import BaseCommand
from config.utils import *


class Command(BaseCommand):
    # def add_arguments(self, parser):
    #     parser.add_argument("poll_ids", nargs="+", type=int)

    def handle(self, *args, **options):
        print("Getting Historic Prices")
        prices = pd.DataFrame(list(PriceHistory.objects.all().values()))

        start = pd.Timestamp("2023-07-01", tz="GB")
        if len(prices) > 0:
            prices.index = pd.to_datetime(prices["date_time"])
            prices = prices.sort_index()
            prices.index = prices.index.tz_convert("GB")
            prices.drop(["id", "date_time"], axis=1, inplace=True)
            start = prices.index[-1] + pd.Timedelta("30min")

        agile = get_agile(start=start)

        day_ahead = day_ahead_to_agile(agile, reverse=True)

        new_prices = pd.concat([day_ahead, agile], axis=1)
        if len(prices) > 0:
            new_prices = new_prices[new_prices.index > prices.index[-1]]
        print(new_prices)
        if len(new_prices) > 0:
            print(new_prices)
            try:
                df_to_Model(new_prices, PriceHistory)
            except:
                pass
