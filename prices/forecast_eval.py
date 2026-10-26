"""Forecast evaluation on user-facing metrics, and out-of-sample spread calibration.

Mean error rewards hedging towards the middle, which is exactly the tail
compression the site suffers from, so alongside MAE/RMSE this scores what Agile
users act on: the cheapest and priciest windows of each day, negative prices,
the within-day ordering of slots, and error in the tails of the outturn.

Calibration remaps a forecast's distribution onto the outturn's, per lead-time
bucket, fitted for each run only on (forecast, outturn) pairs from earlier runs
whose outturn was already known when that run was made. No model is retrained,
so unlike the ensemble retrain backtests this is a faithful replay.

Frames are long: one row per (run, slot) with columns run, created_at,
date_time (UTC), lead_days, actual and actual_da, plus one Agile G column
(p/kWh) per forecast being compared. Calibration works in day-ahead space so
the peak adder, shift and VAT are not stretched along with the price.
"""

import numpy as np
import pandas as pd

from config.utils import day_ahead_to_agile
from prices.models import AgileData, Forecasts, PriceHistory

REGION = "G"
SLOT = pd.Timedelta("30min")
LEAD_BUCKETS = (0, 1, 2, 4, 7, np.inf)  # days ahead
CHEAP_WINDOW_SLOTS = 8  # 4 h
PEAK_WINDOW_SLOTS = 6  # 3 h


def to_agile(day_ahead, date_time):
    s = pd.Series(np.asarray(day_ahead, dtype=float), index=pd.DatetimeIndex(date_time))
    return day_ahead_to_agile(s, region=REGION).to_numpy()


def to_day_ahead(agile, date_time):
    s = pd.Series(np.asarray(agile, dtype=float), index=pd.DatetimeIndex(date_time))
    return day_ahead_to_agile(s, reverse=True, region=REGION).to_numpy()


def load_ensemble_frame(since=None):
    """Every stored run's region-G Agile forecast against the outturn.

    Slots a run copied from already-published prices are dropped: the update
    splices known prices in verbatim, so they would score as perfect forecasts.
    """
    runs = Forecasts.objects.order_by("created_at")
    if since:
        runs = runs.filter(created_at__gte=pd.Timestamp(since, tz="GB"))
    rows = AgileData.objects.filter(region=REGION, forecast__in=runs).values_list(
        "forecast__name", "forecast__created_at", "date_time", "agile_pred"
    )
    df = pd.DataFrame(list(rows), columns=["run", "created_at", "date_time", "ensemble"])
    prices = pd.DataFrame(
        list(PriceHistory.objects.values_list("date_time", "agile", "day_ahead")),
        columns=["date_time", "actual", "actual_da"],
    )
    for frame in (df, prices):
        frame["date_time"] = pd.to_datetime(frame["date_time"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df = df.merge(prices, on="date_time", how="left")

    known = (df["ensemble"] - df["actual"]).abs() <= 0.01
    # First slot of each run that it actually forecast; everything from there on counts.
    first_forecast = df.loc[~known].groupby("run")["date_time"].min()
    df = df[df["date_time"] >= df["run"].map(first_forecast)]
    df = df.dropna(subset=["actual"])
    df["lead_days"] = (df["date_time"] - df["created_at"]).dt.total_seconds() / 86400
    return df.reset_index(drop=True)


def add_backtest_column(df, csv_path, name, value_col="timestral"):
    """Join a timestral_backtest CSV (day-ahead, per run and slot) as an Agile G column."""
    bt = pd.read_csv(csv_path, usecols=["date_time", "run", value_col])
    bt["date_time"] = pd.to_datetime(bt["date_time"], utc=True)
    bt[name] = to_agile(bt[value_col], bt["date_time"])
    return df.merge(bt[["run", "date_time", name]], on=["run", "date_time"], how="left")


def lead_bucket(lead_days):
    return pd.cut(lead_days, LEAD_BUCKETS, right=False, labels=False)


def _stretch(x_hist, y_hist, x):
    sd_x = x_hist.std()
    if not sd_x > 0:
        return x
    return y_hist.mean() + (x - x_hist.mean()) * (y_hist.std() / sd_x)


def _quantile_map(x_hist, y_hist, x):
    """Empirical quantile mapping: y = Q_y(F_x(x)), linear beyond the fitted range."""
    probs = np.linspace(0.005, 0.995, 199)
    qx, qy = np.quantile(x_hist, probs), np.quantile(y_hist, probs)
    qx = np.maximum.accumulate(qx + np.arange(len(qx)) * 1e-9)  # strictly increasing for interp
    out = np.interp(x, qx, qy)
    out = np.where(x < qx[0], qy[0] + (x - qx[0]), out)
    return np.where(x > qx[-1], qy[-1] + (x - qx[-1]), out)


CALIBRATORS = {"stretch": _stretch, "qmap": _quantile_map}


def calibrate(df, col, method="stretch", window_days=28, min_pairs=500):
    """Out-of-sample calibrated copy of an Agile G forecast column.

    For each run, fit per lead bucket on earlier runs' slots that had settled by
    the time the run was made (within the trailing window), then apply. Runs
    without enough history keep their raw forecast.
    """
    fit = CALIBRATORS[method]
    da = to_day_ahead(df[col], df["date_time"])
    bucket = lead_bucket(df["lead_days"]).to_numpy()
    created = df["created_at"].to_numpy()
    settled = (df["date_time"] + SLOT).to_numpy()
    valid = ~np.isnan(da)
    out = da.copy()
    window = np.timedelta64(int(window_days * 86400), "s")

    for run_created in np.unique(created[valid]):
        this_run = valid & (created == run_created)
        history = valid & (created < run_created) & (created >= run_created - window) & (settled <= run_created)
        for b in np.unique(bucket[this_run]):
            hist = history & (bucket == b)
            if hist.sum() < min_pairs:
                continue
            rows = this_run & (bucket == b)
            out[rows] = fit(da[hist], df["actual_da"].to_numpy()[hist], da[rows])
    return to_agile(out, df["date_time"])


def _window_means(values, width):
    return np.convolve(values, np.ones(width) / width, mode="valid")


def daily_window_scores(df, col):
    """Per (run, GB day) with a full day of slots: cheap/peak window regret and rank correlation."""
    day = df["date_time"].dt.tz_convert("GB").dt.normalize()
    records = []
    for (_, _), g in df.assign(day=day).groupby(["run", "day"]):
        if len(g) != 48 or g[col].isna().any():
            continue
        g = g.sort_values("date_time")
        actual, fc = g["actual"].to_numpy(), g[col].to_numpy()
        a_cheap, f_cheap = _window_means(actual, CHEAP_WINDOW_SLOTS), _window_means(fc, CHEAP_WINDOW_SLOTS)
        a_peak, f_peak = _window_means(actual, PEAK_WINDOW_SLOTS), _window_means(fc, PEAK_WINDOW_SLOTS)
        records.append(
            {
                "cheap_regret": a_cheap[f_cheap.argmin()] - a_cheap.min(),
                "cheap_hit": abs(int(f_cheap.argmin()) - int(a_cheap.argmin())) <= 2,  # within 1 h
                "peak_regret": a_peak.max() - a_peak[f_peak.argmax()],
                "rank_corr": pd.Series(fc).rank().corr(pd.Series(actual).rank()),
            }
        )
    return pd.DataFrame(records)


def score(df, cols):
    """User-facing scorecard (Agile G, p/kWh) for each column, on rows where all are present."""
    df = df.dropna(subset=list(cols))
    lo, hi = df["actual"].quantile([0.1, 0.9])
    bottom, top, negative = df["actual"] <= lo, df["actual"] >= hi, df["actual"] < 0
    table = {}
    for col in cols:
        err = df[col] - df["actual"]
        windows = daily_window_scores(df, col)
        predicted_negative = df[col] < 0
        table[col] = {
            "MAE": err.abs().mean(),
            "RMSE": np.sqrt((err**2).mean()),
            "bias": err.mean(),
            "sd ratio": df[col].std() / df["actual"].std(),
            "MAE bottom 10%": err[bottom].abs().mean(),
            "MAE top 10%": err[top].abs().mean(),
            "neg recall": (predicted_negative & negative).sum() / max(negative.sum(), 1),
            "neg precision": (predicted_negative & negative).sum() / max(predicted_negative.sum(), 1),
            "cheap 4h regret": windows["cheap_regret"].mean(),
            "cheap 4h hit": windows["cheap_hit"].mean(),
            "peak 3h regret": windows["peak_regret"].mean(),
            "day rank corr": windows["rank_corr"].mean(),
        }
    table = pd.DataFrame(table)
    table.attrs.update(slots=len(df), runs=df["run"].nunique(), negative_slots=int(negative.sum()))
    return table
