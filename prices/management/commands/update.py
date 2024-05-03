import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
import numpy as np

from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *

DAYS_TO_INCLUDE = 7


class Command(BaseCommand):

    def handle(self, *args, **options):
        # Clean any invalid forecasts
        for f in Forecasts.objects.all():
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            print(f.name, q.count(), a.count())
            if q.count() < 600 or a.count() < 8000:
                f.delete()

        new_name = pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M")
        if new_name not in [f.name for f in Forecasts.objects.all()]:
            base_forecasts = Forecasts.objects.all()
            this_forecast = Forecasts(name=new_name)
            this_forecast.save()

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

            if len(fc) > 0:

                X = hist.iloc[-48 * 56 :]
                y = prices["day_ahead"].loc[X.index]

                cols = X.columns

                for i in range(10):
                    X1 = X.copy()
                    y1 = y.copy()
                    for f in base_forecasts:
                        days_since_forecast = (pd.Timestamp.now(tz="GB") - f.created_at).days
                        if days_since_forecast < 14:

                            # if f != this_forecast and days_since_forecast < 14:
                            df = get_forecast_from_model(forecast=f).loc[: prices.index[-1]]
                            # print(prices.loc[df.index])
                            # print(df.columns)

                            if len(df) > 0:
                                rng = np.random.default_rng()
                                max_len = DAYS_TO_INCLUDE * 48
                                samples = rng.triangular(0, 0, max_len, max_len).astype(int)
                                samples = samples[samples < len(df)]
                                print(
                                    f"{f.id:3d}:, {df.index[0].strftime('%d-%b %H:%M')} - {df.index[-1].strftime('%d-%b %H:%M')}  Length: {len(df.iloc[samples]):3d} Oversampling:{len(df.iloc[samples])/len(df) * 100:0.0f}%"
                                )

                                df = df.iloc[samples]

                                X1 = pd.concat([X1, df[cols]])
                                y1 = pd.concat([y1, prices["day_ahead"].loc[df.index]])

                    model = xg.XGBRegressor(
                        objective="reg:squarederror",
                        booster="dart",
                        # max_depth=0,
                        gamma=0.3,
                        eval_metric="rmse",
                    )

                    model.fit(X1, y1, verbose=True)
                    model_day_ahead = pd.Series(index=y1.index, data=model.predict(X1))

                    model_agile = day_ahead_to_agile(model_day_ahead)
                    rmse = MSE(model_agile, prices["agile"].loc[X1.index]) ** 0.5

                    print(f"RMS Error: {rmse: 0.2f} p/kWh")

                    fc[f"day_ahead_{i}"] = model.predict(fc[cols])

                day_ahead_cols = [f"day_ahead_{i}" for i in range(10)]
                fc["day_ahead"] = fc[day_ahead_cols].mean(axis=1)

                fc["stdev"] = fc[day_ahead_cols].std(axis=1)
                day_ahead_cols += ["stdev"]
                # print(fc[day_ahead_cols + ["day_ahead"]].to_string())

                fc = fc.drop(day_ahead_cols, axis=1)

                ag = pd.concat(
                    [
                        pd.DataFrame(
                            index=fc.index,
                            data={
                                "region": region,
                                "agile_pred": day_ahead_to_agile(fc["day_ahead"], region=region)
                                .astype(float)
                                .round(2),
                            },
                        )
                        for region in regions
                    ]
                )

                fc.drop(["time", "day_of_week"], axis=1, inplace=True)

                fc["forecast"] = this_forecast
                ag["forecast"] = this_forecast
                df_to_Model(fc, ForecastData)
                df_to_Model(ag, AgileData)

            else:
                this_forecast.delete()

        for f in Forecasts.objects.all():
            print(f"{f.id:4d}: {f.name}")

        print(f"Lengths: History: {(len(X) / len(X1)*100):0.1f}%")
        print(f"       Forecasts: {((len(X1) - len(X))/len(X1)*100):0.1f}%")
