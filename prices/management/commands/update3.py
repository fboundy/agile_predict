import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split

import numpy as np

from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *
from config.settings import GLOBAL_SETTINGS

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50
MIN_HIST = 7
MAX_HIST = 28
MAX_TEST_X = 20000


def kde_quantiles(kde, dt, pred, quantiles={"low": 0.1, "mid": 0.5, "high": 0.9}, lim=(0, 150)):
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
            if len(c[c < quantiles[q]]) > 0:
                idx = c[c < quantiles[q]].index[-1]
                results[q] += [(quantiles[q] - c[idx]) / (c[idx + 1] - c[idx]) + idx]
            else:
                results[q] += [np.nan]
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

        parser.add_argument(
            "--no_ranges",
            action="store_true",
        )

    def handle(self, *args, **options):
        debug = options.get("debug", False)

        no_ranges = options.get("no_ranges", False)

        use_gb60 = options.get("gb60", False)

        drop_cols = ["emb_wind"]
        if options.get("no_day_of_week", False):
            drop_cols += ["day_of_week"]

        drop_last = int(options.get("drop_last", 0) or 0)

        if options.get("ignore_forecast", []) is None:
            ignore_forecast = []
        else:
            ignore_forecast = [int(x) for x in options.get("ignore_forecast", [])]

        # Clean any invalid forecasts
        for f in Forecasts.objects.all():
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            if debug:
                print(f.id, f.name, q.count(), a.count())
            if q.count() < 600 or a.count() < 8000:
                f.delete()

        # hist = get_history_from_model()

        # if debug:
        #     print("Getting history from model")
        #     print("Database History:")
        #     print(hist)
        # start = pd.Timestamp("2023-07-01", tz="GB")
        # if len(hist) > 48:
        #     hist = hist.iloc[:-48]
        #     start = hist.index[-1] + pd.Timedelta("30min")
        #     # for h in History.objects.filter(date_time__gte=start):
        #     #     h.delete()
        # else:
        #     hist = pd.DataFrame()

        # if debug:
        #     print(f"New data from {start.strftime(TIME_FORMAT)}:")

        # print("Loading new history:")
        # new_hist, missing_hist = get_latest_history(start=start)

        # if len(new_hist) > 0:
        #     if debug:
        #         print(new_hist)
        #     df_to_Model(new_hist, History, update=True)

        # else:
        #     print("None")

        # hist = pd.concat([hist, new_hist]).sort_index()

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

        agile_end = prices.index[-1]
        gb60 = get_gb60()

        if debug:
            print(f"GB60:\n{gb60}")

        gb60 = gb60.resample("30min").ffill().loc[agile_end + pd.Timedelta("30min") :]

        if len(gb60) > 0:
            gb60 = gb60.reindex(
                pd.date_range(gb60.index[0], gb60.index[-1] + pd.Timedelta("30min"), freq="30min")
            ).ffill()
            gb60 = pd.concat([gb60, day_ahead_to_agile(gb60)], axis=1).set_axis(["day_ahead", "agile"], axis=1)
            prices = pd.concat([prices, gb60]).sort_index()

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
            last_forecasts = {
                forecast.created_at.date(): forecast.id for forecast in base_forecasts.order_by("created_at")
            }

            base_forecasts = base_forecasts.filter(id__in=[last_forecasts[k] for k in last_forecasts])

            if debug:
                print("Getting latest Forecast")

            fc, missing_fc = get_latest_forecast()

            if len(missing_fc) > 0:
                print(">>> ERROR: Unable to run forecast due to missing columns:", end="")
                for c in missing_fc:
                    print(c, end="")

            else:
                if debug:
                    print(fc)

                if len(fc) > 0:
                    fd = pd.DataFrame(list(ForecastData.objects.exclude(forecast_id__in=ignore_forecast).values()))

                    ff = (
                        pd.DataFrame(list(Forecasts.objects.exclude(id__in=ignore_forecast).values()))
                        .set_index("id")
                        .sort_index()
                    )

                    ff["date"] = ff["created_at"].dt.tz_convert("GB").dt.normalize()
                    ff["ag_start"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=22)
                    ff["ag_end"] = ff["created_at"].dt.normalize() + pd.Timedelta(hours=46)

                    # Only train on the forecasts closest to 16:15
                    ff["dt1600"] = (
                        (ff["date"] + pd.Timedelta(hours=16, minutes=15) - ff["created_at"].dt.tz_convert("GB"))
                        .dt.total_seconds()
                        .abs()
                    )
                    ff_train = (
                        ff.sort_values("dt1600").drop_duplicates("date").sort_index().drop(["date", "dt1600"], axis=1)
                    )

                    # df is the full dataset
                    df = (
                        (fd.merge(ff, right_index=True, left_on="forecast_id"))
                        .set_index("date_time")
                        .drop("day_ahead", axis=1)
                    )

                    df["dow"] = df.index.day_of_week
                    df["weekend"] = (df.index.day_of_week >= 5).astype(int)
                    df["time"] = df.index.tz_convert("GB").hour + df.index.minute / 60
                    df["days_ago"] = (pd.Timestamp.now(tz="UTC") - df["created_at"]).dt.total_seconds() / 3600 / 24
                    df["dt"] = (df.index - df["created_at"]).dt.total_seconds() / 3600 / 24
                    df["peak"] = ((df["time"] >= 16) & (df["time"] < 19)).astype(float)

                    max_days = 60

                    features = [
                        "bm_wind",
                        "solar",
                        "demand",
                        # "time",
                        "peak",
                        "days_ago",
                        # "dow",
                        "wind_10m",
                        "weekend",
                    ]

                    # Only use the forecasts closest to 16:15 for training
                    train_X = df[df["forecast_id"].isin(ff_train.index)]
                    train_X = train_X[train_X["days_ago"] < max_days]

                    # Only train on the next agile prices that are set from the pm auction
                    train_X = train_X[(train_X.index >= train_X["ag_start"]) & (train_X.index < train_X["ag_end"])][
                        features
                    ]

                    # Get the prices to match the forecast
                    train_X = train_X.merge(prices["day_ahead"], left_index=True, right_index=True)
                    train_y = train_X.pop("day_ahead")
                    sample_weights = ((np.log10((train_y - train_y.mean()).abs() + 10) * 5) - 4).round(0)

                    xg_model = xg.XGBRegressor(
                        objective="reg:squarederror",
                        booster="dart",
                        gamma=0.2,
                        subsample=1.0,
                        n_estimators=200,
                        max_depth=10,
                        colsample_bytree=1,
                    )

                    xg_model.fit(train_X, train_y, sample_weight=sample_weights, verbose=True)

                    # Drop the training data set
                    test_X = df[~df["forecast_id"].isin(ff_train.index)]

                    # Drop any data which is actual ir dt < 0
                    test_X = test_X[test_X.index > test_X["ag_start"]]

                    # Drop the old data
                    test_X = test_X[test_X["days_ago"] < max_days]

                    test_X = test_X.merge(prices["day_ahead"], left_index=True, right_index=True)
                    test_y = test_X["day_ahead"]

                    if len(test_X) > MAX_TEST_X:
                        _, test_X, _, _ = train_test_split(test_X, test_y, test_size=MAX_TEST_X)

                    if debug:
                        print(f"len(ff)      : {len(ff)}")
                        print(f"len(ff_train): {len(ff_train)}")
                        print(f"len(train_X) : {len(train_X)}")
                        print(f"len(test_X)  : {len(test_X)}")

                        print(f"Earliest ff   : {ff.index.min()}")
                        print(f"Latest ff     : {ff.index.max()}")
                        print(f"Earliest ff_t : {ff_train.index.min()}")
                        print(f"Latest ff_t   : {ff_train.index.max()}")

                        print("train_cols:")
                        for col in train_X.columns:
                            print(
                                f"  {col:16s}:  {train_X[col].min():10.2f} {train_X[col].mean():10.2f} {train_X[col].max():10.2f}"
                            )

                    results = test_X[["dt", "day_ahead"]].copy()
                    results["pred"] = xg_model.predict(test_X[features])

                    # print("Results:")
                    # print(results.to_string())

                    fc["weekend"] = (fc.index.day_of_week >= 5).astype(int)
                    fc["days_ago"] = 0
                    fc["time"] = fc.index.tz_convert("GB").hour + fc.index.minute / 60
                    fc["day_ahead"] = xg_model.predict(fc.drop("emb_wind", axis=1).reindex(train_X.columns, axis=1))
                    fc["dt"] = (fc.index - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400

                    if (len(test_X) > 10) and (not no_ranges):

                        kde = KernelDensity()
                        kde.fit(results[["dt", "pred", "day_ahead"]].to_numpy())

                        xlim = (
                            np.floor(results[["pred", "day_ahead"]].min(axis=1).min() / 11) * 10,
                            np.ceil(results[["pred", "day_ahead"]].max(axis=1).max() / 9) * 10,
                        )

                        fc = pd.concat(
                            [
                                fc,
                                pd.DataFrame(
                                    index=fc.index,
                                    data=kde_quantiles(
                                        kde,
                                        fc["dt"].to_list(),
                                        fc["day_ahead"].to_list(),
                                        lim=xlim,
                                        quantiles={"day_ahead_low": 0.1, "day_ahead_high": 0.9},
                                    ),
                                ),
                            ],
                            axis=1,
                        )

                        for case in ["low", "high"]:
                            fc[f"day_ahead_{case}"] = (
                                fc[f"day_ahead_{case}"].rolling(3, center=True).mean().bfill().ffill()
                            )

                        fc["day_ahead_low"] = fc[["day_ahead", "day_ahead_low"]].min(axis=1)
                        fc["day_ahead_high"] = fc[["day_ahead", "day_ahead_high"]].max(axis=1)

                    else:
                        fc["day_ahead_low"] = fc["day_ahead"] * 0.9
                        fc["day_ahead_high"] = fc["day_ahead"] * 1.1

                    if debug:
                        print(f"Forecast from {fc.index[0]} tp {fc.index[-1]}")
                        print(f"Agile to      {agile_end}")
                        if len(gb60) > 0:
                            print(f"GB60 to       {prices.index[-1]}")

                    idx0 = pd.date_range(fc.index[0], agile_end, freq="30min")

                    if len(gb60) > 0:
                        idx1 = pd.date_range(gb60.index[0], prices.index[-1], freq="30min")
                        idx2 = fc.index.difference(idx0.union(idx1))
                    else:
                        idx1 = pd.Index([])
                        idx2 = fc.index.difference(idx0)

                    fc = fc.astype(float)

                    x = pd.concat(
                        [
                            pd.DataFrame(index=idx0, data={"mult": 0, "shift": 1}),
                            pd.DataFrame(index=idx1, data={"mult": 0, "shift": 5}),
                            pd.DataFrame(index=idx2, data={"mult": 1, "shift": 0}),
                        ],
                    )

                    x = pd.concat([x, prices.reindex(x.index).fillna(0)], axis=1)

                    if debug:
                        print(fc.iloc[:50].to_string())

                    fc["day_ahead"] = fc["day_ahead"] * x["mult"] + x["day_ahead"] * (1 - x["mult"])
                    fc["day_ahead_low"] = (
                        fc["day_ahead_low"] * x["mult"] + x["day_ahead"] * (1 - x["mult"]) - x["shift"]
                    )
                    fc["day_ahead_high"] = (
                        fc["day_ahead_high"] * x["mult"] + x["day_ahead"] * (1 - x["mult"]) + x["shift"]
                    )

                    if debug:
                        print(
                            pd.concat([x, fc[["day_ahead", "day_ahead_low", "day_ahead_high"]]], axis=1)
                            .iloc[:50]
                            .to_string()
                        )

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
                    fc = fc[list(fd.columns)[3:]]

                    # print(fc.columns)
                    # print(ag.columns)

                    this_forecast = Forecasts(name=new_name)
                    this_forecast.save()
                    fc["forecast"] = this_forecast
                    ag["forecast"] = this_forecast
                    df_to_Model(fc, ForecastData)
                    df_to_Model(ag, AgileData)

        if debug:
            for f in Forecasts.objects.all().order_by("-created_at"):
                print(f"{f.id:4d}: {f.name}")
        else:
            try:
                print(f"\n\nAdded Forecast: {this_forecast.id:>4d}: {this_forecast.name}")
            except:
                print("No forecast added")
