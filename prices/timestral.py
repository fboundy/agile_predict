"""Timestral zero-shot day-ahead forecast, run alongside the main ensemble on dev.

Timestral (Mistral, Oct 2026) is a pretrained time-series foundation model: it is
not trained on our data, only conditioned on it at inference time. The target
context is the day-ahead price history (plus GB60 where already published), and
the covariates are the same upstream features the ensemble uses. Over the context
those come from the shortest-lead stored forecast for each slot, and over the
horizon from the latest stored forecast run.

Output is stored as an ExternalForecast source at region G, so it gets the same
chart overlay, retention and accuracy-history comparison as AgileForecast/X2R.

torch and timestral are dev-only dependencies (requirements-timestral.txt) and
are imported lazily so the module can be loaded where they are absent.
"""

import logging
import time

import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone

from config.utils import day_ahead_to_agile, get_gb60
from prices.external_forecasts import _save_rows
from prices.forecast_features import _bank_holiday_series
from prices.models import ExternalForecast, ForecastData, Forecasts, PriceHistory

logger = logging.getLogger(__name__)

REGION = "G"
FREQ = "30min"
COVARIATES = ["solar", "emb_wind", "bm_wind", "demand", "dispatchable_capacity", "temp_2m", "wind_10m"]
# Quantiles stored as agile_low / agile_high, matching the ensemble's p10/p90 band.
LOW_Q, HIGH_Q = 0.1, 0.9


def known_day_ahead(start, before=None):
    """Day-ahead prices from start (UTC) onward: stored history, then GB60 beyond it.

    With ``before`` (a backtest), only stored prices for slots before it are used
    and GB60 is not fetched, since it is only available for the current day.
    """
    rows = PriceHistory.objects.filter(date_time__gte=start)
    if before is not None:
        rows = rows.filter(date_time__lt=before)
    prices = pd.Series(dict(rows.order_by("date_time").values_list("date_time", "day_ahead")), dtype=float)
    prices.index = pd.DatetimeIndex(prices.index).tz_convert("UTC")

    gb60 = get_gb60() if before is None else None
    if gb60 is not None and len(gb60) > 0:
        gb60 = gb60.resample(FREQ).ffill()
        gb60 = gb60.reindex(pd.date_range(gb60.index[0], gb60.index[-1] + pd.Timedelta(FREQ), freq=FREQ)).ffill()
        gb60.index = gb60.index.tz_convert("UTC")
        gb60 = gb60[gb60.index > prices.index[-1]]
        prices = pd.concat([prices, gb60])
    return prices.sort_index()


def stitched_covariates(start, as_of=None):
    """Covariates per slot from the most recent stored forecast issued before it.

    Over the past this is the shortest-lead (nearest-to-actual) value; over the
    future it is the latest run's forecast, since that run is the newest issuer.
    """
    rows = ForecastData.objects.filter(date_time__gte=start)
    if as_of is not None:
        rows = rows.filter(forecast__created_at__lte=as_of)
    rows = rows.values_list("forecast__created_at", "date_time", *COVARIATES)
    df = pd.DataFrame(list(rows), columns=["created_at", "date_time", *COVARIATES])
    if df.empty:
        return df
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    # A run's rows start slightly before it was created; anything older is stale backfill.
    df = df[df["date_time"] >= df["created_at"] - pd.Timedelta("1h")]
    df = df.sort_values("created_at").drop_duplicates("date_time", keep="last")
    return df.set_index("date_time").sort_index()[COVARIATES].astype(float)


def calendar_covariates(index):
    """Known-future calendar rows. All-0/1 rows are treated as indicators by Timestral."""
    gb = index.tz_convert("GB")
    hour = gb.hour + gb.minute / 60
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "peak": ((hour >= 16) & (hour < 19)).astype(float),
            "non_working": ((gb.day_of_week >= 5) | (_bank_holiday_series(index).to_numpy() == 1)).astype(float),
        },
        index=index,
    )


def build_inputs(context_days, now=None, run=None, prices_before=None):
    """Return (index, target, covariates, horizon_start) on a common 30-min UTC grid.

    ``run`` (a Forecasts row) and ``prices_before`` reconstruct what was known when
    that run was made, for backtesting; by default the latest run and all prices.
    """
    as_of = run.created_at if run is not None else None
    now = pd.Timestamp(as_of) if as_of is not None else (now or pd.Timestamp.now(tz="UTC"))
    start = (now - pd.Timedelta(days=context_days)).floor(FREQ)

    latest = run or Forecasts.objects.order_by("-created_at").first()
    if latest is None:
        raise ValueError("No stored forecast run to take horizon covariates from")
    horizon_end = pd.Timestamp(ForecastData.objects.filter(forecast=latest).latest("date_time").date_time).tz_convert("UTC")

    prices = known_day_ahead(start, before=prices_before)
    horizon_start = prices.index[-1] + pd.Timedelta(FREQ)
    if horizon_end < horizon_start:
        raise ValueError(f"Latest forecast run ends {horizon_end}, before the first unpriced slot {horizon_start}")

    index = pd.date_range(start, horizon_end, freq=FREQ)
    target = prices.reindex(index)
    target[index >= horizon_start] = np.nan
    covariates = pd.concat([stitched_covariates(start, as_of).reindex(index), calendar_covariates(index)], axis=1)
    return index, target, covariates, horizon_start


def load_forecaster(model_name=None):
    from timestral import TimestralForecaster

    return TimestralForecaster.from_pretrained(model_name or settings.TIMESTRAL_MODEL)


def forecast_day_ahead(model_name=None, context_days=None, now=None, run=None, prices_before=None, forecaster=None):
    """Run Timestral and return a DataFrame of day-ahead median/low/high over the horizon."""
    model_name = model_name or settings.TIMESTRAL_MODEL
    context_days = context_days or settings.TIMESTRAL_CONTEXT_DAYS

    index, target, covariates, horizon_start = build_inputs(context_days, now=now, run=run, prices_before=prices_before)
    is_context = index < horizon_start
    context = target[is_context].to_numpy()
    horizon = int((~is_context).sum())
    logger.info(
        "Timestral inputs: model=%s context=%d slots (%d priced) horizon=%d slots from %s covariates=%s",
        model_name, len(context), int(np.isfinite(context).sum()), horizon, horizon_start, list(covariates.columns),
    )

    started = time.monotonic()
    forecaster = forecaster or load_forecaster(model_name)
    result = forecaster.forecast(
        context[None, None, :],
        prediction_length=horizon,
        known_covariates=covariates.to_numpy().T[None, :, :],
    )
    logger.info("Timestral forecast elapsed_seconds=%.1f", time.monotonic() - started)

    horizon_index = index[~is_context]
    return pd.DataFrame(
        {
            "day_ahead": np.asarray(result.median)[0, 0],
            "day_ahead_low": np.asarray(result.quantile(LOW_Q))[0, 0],
            "day_ahead_high": np.asarray(result.quantile(HIGH_Q))[0, 0],
        },
        index=horizon_index.tz_convert("GB"),
    )


def run_and_store(model_name=None, context_days=None):
    """Forecast and save as the region-G Timestral ExternalForecast snapshot. Returns row count."""
    model_name = model_name or settings.TIMESTRAL_MODEL
    created_at = timezone.now()
    fc = forecast_day_ahead(model_name, context_days, now=pd.Timestamp(created_at))
    agile = {col: day_ahead_to_agile(fc[f"day_ahead{suffix}"], region=REGION).astype(float).round(2)
             for col, suffix in (("agile_pred", ""), ("agile_low", "_low"), ("agile_high", "_high"))}
    rows = [
        {
            "date_time": ts.to_pydatetime(),
            "agile_pred": float(agile["agile_pred"].loc[ts]),
            "agile_low": float(agile["agile_low"].loc[ts]),
            "agile_high": float(agile["agile_high"].loc[ts]),
        }
        for ts in fc.index
    ]
    name = f"{model_name.split('/')[-1]} {timezone.localtime(created_at):%Y-%m-%d %H:%M}"
    count = _save_rows(ExternalForecast.SOURCE_TIMESTRAL, REGION, name, created_at, rows)
    logger.info("Saved Timestral forecast %s rows=%d", name, count)
    return count
