import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neighbors import KernelDensity

import numpy as np

from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData, Nordpool, UpdateErrors

from config.utils import *
from config.settings import GLOBAL_SETTINGS

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50
MIN_HIST = 7
MAX_HIST = 28


def kde_quantiles(kde, dt, pred, quantiles={"low": 0.1, "mid":0.5, "high":0.9}, lim=(0, 150)):
    if not isinstance(dt, list):
        dt = [dt]
    if not isinstance(pred, list):
        pred = [pred]

    results = {q: [] for q in quantiles}
    for dt1, pred1 in zip(dt, pred):
        x = np.array([[dt1, pred1, p] for p in range(int(lim[0]), int(lim[1]))])
        c = pd.Series(index=x[:, 2], data=np.exp(kde.score_samples(x)).cumsum())
        c /= c.iloc[-1]

        for q in quantiles:
            idx = c[c < quantiles[q]].index[-1]
            results[q] += [(quantiles[q] - c[idx]) / (c[idx + 1] - c[idx]) + idx]

    return results

class Command(BaseCommand):
    def add_arguments(self, parser):
        # Positional arguments
        # parser.add_argument("poll_ids", nargs="+", type=int)

        # Named (optional) arguments
        parser.add_argument(
            "--debug",
            action="store_true",
        )

        parser.add_argument(
            "--no_day_of_week",
            action="store_true",
        )

        parser.add_argument(
            "--nordpool",
            action="store_true",
        )

        parser.add_argument(
            "--train_frac",
        )

        parser.add_argument(
            "--drop_last",
        )


        parser.add_argument(
            "--ignore_forecast",
            action="append",
        )



    def handle(self, *args, **options):
        debug = options.get("debug", False)

        use_nordpool = options.get("nordpool", False)

        drop_cols = ["emb_wind"]
        if options.get("no_day_of_week", False):
            drop_cols += ["day_of_week"]

        drop_last = int(options.get("drop_last", 0) or 0)

        frac=float( options.get("train_frac", 0.8) or 0.8)

        if options.get("ignore_forecast", []) is None:
            ignore_forecast = []
        else:
            ignore_forecast = [int(x) for x in options.get("ignore_forecast", [])]

        UpdateErrors.objects.all().delete()

        # Clean any invalid forecasts
        for f in Forecasts.objects.all():
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            if debug:
                print(f.id, f.name, q.count(), a.count())
            if q.count() < 600 or a.count() < 8000:
                f.delete()

        hist = get_history_from_model()

        if debug:
            print("Getting history from model")
            print("Database History:")
            print(hist)
        start = pd.Timestamp("2023-07-01", tz="GB")
        if len(hist) > 48:
            hist = hist.iloc[:-48]
            start = hist.index[-1] + pd.Timedelta("30min")
            # for h in History.objects.filter(date_time__gte=start):
            #     h.delete()
        else:
            hist = pd.DataFrame()

        if debug:
            print(f"New data from {start.strftime(TIME_FORMAT)}:")

        print("day_of_weeknloading new history:")
        new_hist, missing_hist = get_latest_history(start=start)

        if len(new_hist) > 0:
            if debug:
                print(new_hist)
            df_to_Model(new_hist, History, update=True)

        else:
            print("None")

        hist = pd.concat([hist, new_hist]).sort_index()

        if debug:
            print("Getting Historic Prices")

        prices, start = model_to_df(PriceHistory)
        agile = get_agile(start=start)

        day_ahead = day_ahead_to_agile(agile, reverse=True)

        new_prices = pd.concat([day_ahead, agile], axis=1)
        if len(prices) > 0:
            new_prices = new_prices[new_prices.index > prices.index[-1]]

        if debug:
            print(new_prices)

        if len(new_prices) > 0:
            print(new_prices)
            df_to_Model(new_prices, PriceHistory)
            prices = pd.concat([prices, new_prices]).sort_index()

        if use_nordpool:
            nordpool = pd.DataFrame(get_nordpool(start=prices.index[-1] + pd.Timedelta("30min"))).set_axis(
                ["day_ahead"], axis=1
            )
            if len(nordpool)>0:
                print(f"Hourly day ahead data used for period: {nordpool.index[0].strftime("%d-%b %H:%M")} - {nordpool.index[-1].strftime("%d-%b %H:%M")}")
            nordpool["agile"] = day_ahead_to_agile(nordpool["day_ahead"])

            if debug:
                print(f"Database prices:\n{prices}")
                print(f"New prices:\n{prices}")
                print(f"Nordpool prices:\n{prices}")

            prices = pd.concat([prices, nordpool]).sort_index()

        if debug:
            print(f"Merged prices:\n{prices}")

        if drop_last > 0:
            print(f"drop_last: {drop_last}")
            print(f"len: {len(prices)} last:{prices.index[-1]}")
            prices = prices.iloc[:-drop_last]
            print(f"len: {len(prices)} last:{prices.index[-1]}")

        new_name = pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M")
        if new_name not in [f.name for f in Forecasts.objects.all()]:
            base_forecasts = Forecasts.objects.exclude(id__in=ignore_forecast).order_by("-created_at")
            last_forecasts ={forecast.created_at.date(): forecast.id for forecast in base_forecasts.order_by("created_at")}

            base_forecasts=base_forecasts.filter(id__in=[last_forecasts[k] for k in last_forecasts])

            if debug:
                print("Getting latest Forecast")
            fc, missing_fc = get_latest_forecast()

            if len(missing_fc) > 0:
                print(">>> ERROR: Unable to run forecast due to missing columns:", end="")
                for c in missing_fc:
                    print(c, end="")
                    obj=UpdateErrors(
                        date_time=pd.Timestamp.now(tz='GB'),
                        type='Forecast',
                        dataset=c,
                    )
                    obj.save()                

            else:
                if debug:
                    print(fc)

                if len(fc) > 0:
                    fd= pd.DataFrame(list(ForecastData.objects.exclude(forecast_id__in=ignore_forecast).values()))
                    ff = pd.DataFrame(list(Forecasts.objects.exclude(id__in=ignore_forecast).values())).set_index("id").sort_index()

                    ff['date']=ff['created_at'].dt.tz_convert('GB').dt.normalize()
                    ff['dt1600']=(ff['date']+pd.Timedelta(hours=15,minutes=45)-ff['created_at'].dt.tz_convert('GB')).dt.total_seconds().abs()
                    ff=ff.sort_values('dt1600').drop_duplicates('date').sort_index().drop(['date', 'dt1600'],axis=1)

                    ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
                    ff["ag_end"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)
                    ff_train = ff.sample(frac=frac)
                    ff_test = ff.drop(ff_train.index)
                    print(f"Training forecasts ({frac*100:0.0f}%): {ff_train.index}")
                    print(f"Testing forecasts: ({100-frac*100:0.0f}%): {ff_test.index}")

                    train_X = (
                        fd.merge(ff_train, right_index=True, left_on="forecast_id")
                        .drop(["id", "forecast_id", "name", "created_at", "day_ahead"], axis=1)
                    )
                    train_X = (
                        train_X[(train_X["date_time"] >= train_X["ag_start"]) & (train_X["date_time"] < train_X["ag_end"])]
                        .groupby("date_time")
                        .last()
                        .drop(["ag_start", "ag_end"], axis=1)
                    )
                    train_X = (
                        train_X.merge(prices['day_ahead'], left_index=True, right_index=True)
                        .drop(["emb_wind"], axis=1)
                    )
                    train_X["day_of_week"] = train_X.index.day_of_week
                    train_X["time"] = train_X.index.tz_convert("GB").hour + train_X.index.minute / 60
                    train_y = train_X.pop("day_ahead")

                    xg_model = xg.XGBRegressor(
                        objective="reg:squarederror",
                        booster="dart",
                        # max_depth=0,
                        gamma=0.3,
                        eval_metric="rmse",
                        n_estimators=100,
                    )

                    xg_model.fit(train_X, train_y, verbose=True)

                    test_X = (
                        fd.merge(ff_test, right_index=True, left_on="forecast_id").drop(
                            ["id", "forecast_id", "name", "day_ahead", "ag_end"], axis=1
                        )
                    ).set_index("date_time")
                    test_X = test_X[test_X.index > test_X["ag_start"]]
                    test_X["day_of_week"] = test_X.index.day_of_week
                    test_X["time"] = test_X.index.tz_convert("GB").hour + test_X.index.minute / 60
                    results = pd.DataFrame(
                        index=test_X.index, data={"dt": (test_X.index - test_X["created_at"]).dt.total_seconds() / 86400}
                    )

                    test_X = test_X.drop(["created_at", "ag_start", "emb_wind"], axis=1)
                    results["pred"] = xg_model.predict(test_X)
                    results = results[results.index <= prices.index[-1]]
                    results = results.merge(prices['day_ahead'], left_index=True, right_index=True)

                    kde = KernelDensity()
                    kde.fit(results[["dt", "pred", "day_ahead"]].to_numpy())

                    xlim = (
                        np.floor(results[["pred", "day_ahead"]].min(axis=1).min() / 11) * 10,
                        np.ceil(results[["pred", "day_ahead"]].max(axis=1).max() / 9) * 10,
                    )

                    fc["day_of_week"] = fc.index.day_of_week
                    fc["time"] = fc.index.tz_convert("GB").hour + fc.index.minute / 60
                    fc["day_ahead"] = xg_model.predict(fc.drop('emb_wind', axis=1).reindex(train_X.columns,axis=1))
                    fc["dt"]= (fc.index - pd.Timestamp.now(tz='UTC')).total_seconds() / 86400

                    fc = pd.concat(
                        [
                            fc,
                            pd.DataFrame(index=fc.index, data=kde_quantiles(kde, fc["dt"].to_list(), fc["day_ahead"].to_list(), lim=xlim,quantiles={"day_ahead_low":0.1,"day_ahead_high":0.9})),
                        ],
                        axis=1,
                    )

                    for case in ["low", "high"]:
                        fc[f"day_ahead_{case}"] = fc[f"day_ahead_{case}"].rolling(3, center=True).mean().bfill().ffill()

                    fc["day_ahead_low"]=fc[['day_ahead', 'day_ahead_low']].min(axis=1)
                    fc["day_ahead_high"]=fc[['day_ahead', 'day_ahead_high']].max(axis=1)
                    # print(fc.to_string())
                    # fc["day_ahead"] = fc[day_ahead_cols].mean(axis=1)
                    # if iters > 9:
                    #     fc["day_ahead_low"] = fc[day_ahead_cols].quantile(0.1, axis=1)
                    #     fc["day_ahead_high"] = fc[day_ahead_cols].quantile(0.9, axis=1)
                    # else:
                    #     fc["day_ahead_low"] = fc[day_ahead_cols].min(axis=1)
                    #     fc["day_ahead_high"] = fc[day_ahead_cols].max(axis=1)

                    ag = pd.concat(
                        [
                            pd.DataFrame(
                                index=fc.index,
                                data={
                                    "region": region,
                                    "agile_pred": day_ahead_to_agile(fc["day_ahead"], region=region)
                                    .astype(float)
                                    .round(2),
                                    "agile_low": day_ahead_to_agile(fc["day_ahead_low"], region=region)
                                    .astype(float)
                                    .round(2),
                                    "agile_high": day_ahead_to_agile(fc["day_ahead_high"], region=region)
                                    .astype(float)
                                    .round(2),
                                },
                            )
                            for region in regions
                        ]
                    )

                    # fc = fc.drop(day_ahead_cols, axis=1)
                    fc.drop(["time", "day_of_week", "day_ahead_low", "day_ahead_high", "dt"], axis=1, inplace=True)

                    # print(fc.columns)
                    # print(ag.columns)

                    this_forecast = Forecasts(name=new_name)
                    this_forecast.save()
                    fc["forecast"] = this_forecast
                    ag["forecast"] = this_forecast
                    df_to_Model(fc, ForecastData)
                    df_to_Model(ag, AgileData)

        if debug:
            for f in Forecasts.objects.all():
                print(f"{f.id:4d}: {f.name}")
        else:
            try:
                print(f"\n\nAdded Forecast: {this_forecast.id:>4d}: {this_forecast.name}")
            except:
                print("No forecast added")
