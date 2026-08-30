"""Post-hoc correction of the day-ahead forecast, from stored forecast/actual pairs.

Motivation and evidence: docs/MODEL_DYNAMIC_RANGE.md. Two separate effects, both
measured on held-out days at the D-1 (24-48 h) vintage:

* **Surplus carries information the model has not absorbed.** Regressing settled
  price on (prediction, surplus, surplus^2) drops RMSE 17.89 -> 16.99, and the
  surplus coefficient is about +0.74 GBP/MWh per GW of residual demand. The
  prediction coefficient stays near 1.0, so surplus is contributing largely
  orthogonal information rather than re-explaining the prediction.
* **Least squares still compresses**, because it fits a conditional mean: the
  fitted values come out at sd ratio ~0.87. Rescaling them about the training mean
  so their spread matches the actuals — RMA's variance restoration, applied to the
  bivariate fit rather than to the prediction alone — lifts negative-price F1 from
  0.312 to 0.407 and cuts RMSE below GBP 25 from 53.17 (raw) to 35.93.

The cost is real and should not be hidden: precision falls from ~0.99 to 0.83, and
the corrected series over-reaches at the extreme low end (minimum around -GBP 38
against an observed minimum of -GBP 25.3).

Fitting vintage. The relationship is fitted on D-1 pairs only, where the inputs are
sharpest and least contaminated by forecast error, and then applied at every
horizon. That follows the owner's framing: there is one response function, and
horizon changes how well its arguments are known rather than the function itself.
It is an assumption, not a measured result — a correction fitted at D-1 may not
transfer to 10 days, and comparing the corrected and raw series over time is
exactly what this is being run to find out.

Nothing here changes `day_ahead`. The corrected series is stored alongside it in
`day_ahead_corrected` so the two can be scored against each other.
"""

import logging

import numpy as np
import pandas as pd

from prices.models import DayAheadCalibration

logger = logging.getLogger(__name__)

# Fit only on pairs at this horizon, in hours. Below ~36 h the pipeline blends GB60
# day-ahead prices into `day_ahead`, so those slots would train the correction on
# Nord Pool pass-through rather than on model output; 24 h is the lower bound the
# evidence was gathered at, and identical pred/actual pairs are dropped as well.
FIT_HORIZON_H = (24.0, 48.0)

# Below this many usable pairs the fit is not attempted: the correction is a
# 4-parameter model and a thin, seasonally narrow sample would do more harm than
# leaving the column null.
MIN_PAIRS = 500

# Guard rails on the corrected output, as multiples of the observed spread of the
# fitting actuals. The correction is linear-ish and unbounded, so a forecast far
# outside the fitted range could otherwise be mapped somewhere absurd.
CLIP_SD = 4.0


def surplus_gw(df):
    """Residual demand in GW: what thermal plant and imports must cover.

    Positive means conventional generation is needed; negative means renewables
    plus nuclear already exceed demand, which is where plunges live.
    """
    if "nuclear" in df.columns:
        nuclear = pd.to_numeric(df["nuclear"], errors="coerce").fillna(0.0)
    else:
        nuclear = 0.0
    return (df["demand"] - df["bm_wind"] - df["emb_wind"] - df["solar"] - nuclear) / 1000.0


def _design(pred, sur):
    pred = np.asarray(pred, dtype=float)
    sur = np.asarray(sur, dtype=float)
    return np.column_stack([np.ones(len(pred)), pred, sur, sur ** 2])


class DayAheadCorrection:
    """Bivariate fit on (prediction, surplus) with the spread restored."""

    def __init__(self, beta, fit_mean, fit_sd, actual_mean, actual_sd, n, lo, hi):
        self.beta = beta
        self.fit_mean = fit_mean
        self.fit_sd = fit_sd
        self.actual_mean = actual_mean
        self.actual_sd = actual_sd
        self.n = n
        self.lo = lo
        self.hi = hi

    def __call__(self, pred, sur):
        raw = _design(pred, sur) @ self.beta
        # Variance restoration: least squares fits the conditional mean and so
        # under-disperses. Rescale about the fitted mean to match the actuals.
        scaled = self.actual_mean + (self.actual_sd / self.fit_sd) * (raw - self.fit_mean)
        return np.clip(scaled, self.lo, self.hi)

    def describe(self):
        b = self.beta
        return (f"n={self.n} intercept={b[0]:+.2f} pred={b[1]:+.4f} "
                f"surplus={b[2]:+.3f} surplus^2={b[3]:+.4f} "
                f"scale={self.actual_sd / self.fit_sd:.3f} "
                f"clip=[{self.lo:.1f}, {self.hi:.1f}]")


def fit_correction(stored, actuals, horizon_h=FIT_HORIZON_H, min_pairs=MIN_PAIRS):
    """Fit from stored forecasts joined to settled prices.

    `stored` needs date_time, created_at, day_ahead and the generation columns;
    `actuals` is a frame indexed by date_time with a day_ahead column. Returns None
    when there is not enough usable data, and the caller then stores nulls.
    """
    if stored is None or len(stored) == 0:
        return None

    df = stored.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["horizon_h"] = (df["date_time"] - df["created_at"]).dt.total_seconds() / 3600.0

    act = actuals["day_ahead"].rename("actual")
    df = df.join(act, on="date_time", how="inner")

    lo_h, hi_h = horizon_h
    df = df[(df["horizon_h"] >= lo_h) & (df["horizon_h"] < hi_h)]
    # Drop GB60 pass-through: those rows would fit the correction to Nord Pool.
    df = df[(df["day_ahead"] - df["actual"]).abs() >= 0.005]
    df["surplus"] = surplus_gw(df)
    df = df.dropna(subset=["day_ahead", "actual", "surplus"])

    if len(df) < min_pairs:
        logger.info("Day-ahead correction: only %s usable pairs (need %s); skipping",
                    len(df), min_pairs)
        return None

    pred = df["day_ahead"].to_numpy(float)
    sur = df["surplus"].to_numpy(float)
    actual = df["actual"].to_numpy(float)

    X = _design(pred, sur)
    beta, *_ = np.linalg.lstsq(X, actual, rcond=None)
    fitted = X @ beta
    if not np.isfinite(beta).all() or fitted.std() <= 0:
        logger.warning("Day-ahead correction: degenerate fit; skipping")
        return None

    a_mean, a_sd = float(actual.mean()), float(actual.std())
    corr = DayAheadCorrection(
        beta=beta,
        fit_mean=float(fitted.mean()), fit_sd=float(fitted.std()),
        actual_mean=a_mean, actual_sd=a_sd, n=len(df),
        lo=a_mean - CLIP_SD * a_sd, hi=a_mean + CLIP_SD * a_sd,
    )
    logger.info("Day-ahead correction fitted: %s", corr.describe())
    return corr


def harvest_calibration(stored, actuals, horizon_h=FIT_HORIZON_H):
    """Record settled D-1 pairs into DayAheadCalibration; returns rows written.

    Idempotent: slots already recorded are skipped, so this can run on every update.
    Must be called BEFORE the forecast purge, while the source rows still exist.
    """
    if stored is None or len(stored) == 0:
        return 0

    df = stored.copy()
    df["date_time"] = pd.to_datetime(df["date_time"], utc=True)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["horizon_h"] = (df["date_time"] - df["created_at"]).dt.total_seconds() / 3600.0

    df = df.join(actuals["day_ahead"].rename("actual"), on="date_time", how="inner")
    lo_h, hi_h = horizon_h
    df = df[(df["horizon_h"] >= lo_h) & (df["horizon_h"] < hi_h)]
    # GB60 pass-through is not model output and must never enter the fit.
    df = df[(df["day_ahead"] - df["actual"]).abs() >= 0.005]
    df["surplus"] = surplus_gw(df)
    df = df.dropna(subset=["day_ahead", "actual", "surplus"])
    if df.empty:
        return 0

    # One row per slot: the freshest run that is still at least `lo_h` ahead.
    #
    # Note this is NOT the 16:15 auction vintage, and an earlier version of this
    # comment wrongly said it was. For a midday slot the 16:15 run on D-1 is only
    # about 21 h ahead, below this window entirely; the 24-48 h band mixes the D-1
    # run for late-evening slots with the D-2 run for early ones. The band is kept
    # because it is what the correction was validated on, and because it excludes the
    # GB60 blend region — but it is a horizon band, not a publication vintage.
    df = df.sort_values("horizon_h").drop_duplicates("date_time", keep="first")

    have = set(DayAheadCalibration.objects.filter(
        date_time__gte=df["date_time"].min()
    ).values_list("date_time", flat=True))
    new = [
        DayAheadCalibration(
            date_time=row.date_time.to_pydatetime(),
            created_at=row.created_at.to_pydatetime(),
            horizon_h=float(row.horizon_h),
            predicted=float(row.day_ahead),
            actual=float(row.actual),
            surplus=float(row.surplus),
        )
        for row in df.itertuples()
        if row.date_time.to_pydatetime() not in have
    ]
    if new:
        DayAheadCalibration.objects.bulk_create(new, batch_size=1000, ignore_conflicts=True)
        logger.info("Day-ahead calibration: recorded %s new settled pairs (%s held)",
                    len(new), DayAheadCalibration.objects.count())
    return len(new)


def fit_from_calibration(min_pairs=MIN_PAIRS):
    """Fit the correction from the retained calibration table."""
    rows = pd.DataFrame(list(DayAheadCalibration.objects.values(
        "predicted", "actual", "surplus")))
    if len(rows) < min_pairs:
        logger.info("Day-ahead correction: calibration table holds %s pairs (need %s); "
                    "skipping", len(rows), min_pairs)
        return None

    pred = rows["predicted"].to_numpy(float)
    sur = rows["surplus"].to_numpy(float)
    actual = rows["actual"].to_numpy(float)
    X = _design(pred, sur)
    beta, *_ = np.linalg.lstsq(X, actual, rcond=None)
    fitted = X @ beta
    if not np.isfinite(beta).all() or fitted.std() <= 0:
        logger.warning("Day-ahead correction: degenerate fit; skipping")
        return None

    a_mean, a_sd = float(actual.mean()), float(actual.std())
    corr = DayAheadCorrection(
        beta=beta,
        fit_mean=float(fitted.mean()), fit_sd=float(fitted.std()),
        actual_mean=a_mean, actual_sd=a_sd, n=len(rows),
        lo=a_mean - CLIP_SD * a_sd, hi=a_mean + CLIP_SD * a_sd,
    )
    logger.info("Day-ahead correction fitted from calibration table: %s", corr.describe())
    return corr
