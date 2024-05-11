import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
import numpy as np

from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData, Nordpool

from config.utils import *

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50


class Command(BaseCommand):

    def handle(self, *args, **options):
        # Clean any invalid forecasts
        print("Getting Historic Prices")

        prices, start = model_to_df(PriceHistory)
        agile = get_agile(start=start)
        day_ahead = day_ahead_to_agile(agile, reverse=True)

        new_prices = pd.concat([day_ahead, agile], axis=1)
        if len(prices) > 0:
            new_prices = new_prices[new_prices.index > prices.index[-1]]
        print(new_prices)
        if len(new_prices) > 0:
            print(new_prices)
            df_to_Model(new_prices, PriceHistory)

        prices = pd.concat([prices, new_prices]).sort_index()

        print("Getting NP from Model")
        nordpool, start = model_to_df(Nordpool)

        new_nordpool = get_nordpool(start)
        nordpool = pd.concat([nordpool, new_nordpool]).sort_index()
        nordpool.name = "Nordpool"

        df = pd.concat([prices["day_ahead"], nordpool], axis=1)
        print(df.dropna().to_string())

        df_to_Model(nordpool, Nordpool)
