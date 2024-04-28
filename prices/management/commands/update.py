import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE

from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *


class Command(BaseCommand):

    def handle(self, *args, **options):

        print("Getting history from model")
        hist = get_history_from_model()

        print("Database History:")
        print(hist)
        start = pd.Timestamp("2023-07-01", tz="GB")
        if len(hist) > 0:
            start = hist.index[-1] + pd.Timedelta("30min")

        print(f"New data from {start.strftime(TIME_FORMAT)}:")

        new_hist = get_latest_history(start=start)
        if len(new_hist) > 0:
            print(new_hist)
            df_to_Model(new_hist, History)

        else:
            print("None")

        hist = pd.concat([hist, new_hist]).sort_index()

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
            df_to_Model(new_prices, PriceHistory)

        prices = pd.concat([prices, new_prices]).sort_index()

        print("Getting latest Forecast")
        fc = get_latest_forecast()
        print(fc)

        X = hist.iloc[-48 * 28 :]
        y = prices["day_ahead"].loc[X.index]

        cols = X.columns

        for f in Forecasts.objects.all():
            df = get_forecast_from_model(forecast=f).loc[: prices.index[-1]]
            print(f"{f.id:3d}:, {df.index[0].strftime('%d-%b %H:%M')} - {df.index[-1].strftime('%d-%b %H:%M')}")
            print(prices.loc[df.index])
            print(df.columns)
            X = pd.concat([X, df[cols]])
            y = pd.concat([y, prices["day_ahead"].loc[df.index]])

        model = xg.XGBRegressor(
            objective="reg:squarederror",
            booster="dart",
            # max_depth=0,
            gamma=0.3,
            eval_metric="rmse",
        )

        model.fit(X, y, verbose=True)
        model_day_ahead = pd.Series(index=y.index, data=model.predict(X))

        model_agile = day_ahead_to_agile(model_day_ahead)
        rmse = MSE(model_agile, prices["agile"].loc[X.index]) ** 0.5

        print(f"RMS Error: {rmse: 0.2f} p/kWh")

        fc["day_ahead"] = model.predict(fc[cols])

        ag = pd.concat(
            [
                pd.DataFrame(
                    index=fc.index,
                    data={
                        "region": region,
                        "agile_pred": day_ahead_to_agile(fc["day_ahead"], region=region).astype(float).round(2),
                    },
                )
                for region in regions
            ]
        )

        # fc["agile_actual"] = prices["agile"].loc[fc.index[0] :]
        # print(fc[["agile_pred", "agile_actual"]])
        # fc.drop(["time", "day_of_week", "day_of_year", "day_ahead"], axis=1, inplace=True)
        fc.drop(["time", "day_of_week"], axis=1, inplace=True)

        f = Forecasts(name=pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M"))

        f.save()

        fc["forecast"] = f
        ag["forecast"] = f
        df_to_Model(fc, ForecastData)
        df_to_Model(ag, AgileData)
        for f in Forecasts.objects.all():
            print(f.name)
