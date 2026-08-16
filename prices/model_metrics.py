"""
Calibration and rare-event detection metrics for day-ahead price forecasts.

Motivation (see docs/MODEL_DYNAMIC_RANGE.md): the ensemble was materially
under-dispersed in the tails for months without that being visible anywhere in the
logs, because the only score reported was `(wMAE + wRMSE)/2` — an aggregate
dominated by the ~80% of slots in the middle of the distribution. A configuration
can win on that score while never predicting a negative price or a spike.

These metrics are the gate. They are deliberately cheap (pure numpy over a
prediction/actual pair) so they can be logged on every run and alongside every
feature-experiment fold.
"""

import numpy as np

# Product-relevant bands, in £/MWh on the model's native (unscaled) day-ahead
# scale. These are the decisions a user actually makes from the forecast:
# "is power going to be free/paid-for", "is it going to be cheap enough to run a
# load", "is it going to be expensive enough to avoid".
PRICE_BANDS = {
    "negative": {"below": 0.0},
    "cheap": {"below": 50.0},
    "expensive": {"above": 180.0},
    "spike": {"above": 250.0},
}

# Slots this many standard deviations from the mean count as "tail" for tail RMSE.
TAIL_Z = 1.5


def _finite_pair(pred, actual):
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    ok = np.isfinite(pred) & np.isfinite(actual)
    return pred[ok], actual[ok]


def detection_metrics(pred, actual, below=None, above=None):
    """Recall/precision/F1 for a one-sided price band.

    Recall answers "of the slots that really were extreme, how many did we
    flag?"; precision answers "of the slots we flagged, how many really were?".
    Both are needed: a model that shifts every prediction downward scores well on
    negative-price recall while being useless.
    """
    pred, actual = _finite_pair(pred, actual)
    if below is not None:
        actual_hit, pred_hit = actual < below, pred < below
    else:
        actual_hit, pred_hit = actual > above, pred > above

    n_actual = int(actual_hit.sum())
    n_flagged = int(pred_hit.sum())
    n_correct = int((actual_hit & pred_hit).sum())

    recall = n_correct / n_actual if n_actual else float("nan")
    precision = n_correct / n_flagged if n_flagged else float("nan")
    if n_actual and n_flagged and (recall + precision) > 0:
        f1 = 2 * recall * precision / (recall + precision)
    else:
        f1 = float("nan")

    return {
        "n_actual": n_actual,
        "n_flagged": n_flagged,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def calibration_metrics(pred, actual):
    """Dispersion and regime bias.

    `sd_ratio` is sd(pred)/sd(actual): 1.0 means the forecast varies as much as
    reality. Note that a *perfectly* calibrated conditional mean should sit at
    sd_ratio = r, not 1.0 — under-dispersion is only a defect when it exceeds
    that. `slope` is the regression actual ~ pred, which should be 1.0 for a
    predictor that is the MSE-optimal linear function of itself; a slope above 1
    means the predictions need stretching and squared error is recoverable.
    """
    pred, actual = _finite_pair(pred, actual)
    out = {
        "n": int(pred.size),
        "rmse": float("nan"),
        "mae": float("nan"),
        "sd_ratio": float("nan"),
        "slope": float("nan"),
        "r": float("nan"),
        "tail_rmse": float("nan"),
        "low_bias": float("nan"),
        "high_bias": float("nan"),
    }
    if pred.size < 2:
        return out

    err = pred - actual
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mae"] = float(np.mean(np.abs(err)))

    sd_actual, sd_pred = float(actual.std()), float(pred.std())
    if sd_actual > 0:
        out["sd_ratio"] = sd_pred / sd_actual
        z = (actual - actual.mean()) / sd_actual
        tail = np.abs(z) > TAIL_Z
        if tail.sum() > 5:
            out["tail_rmse"] = float(np.sqrt(np.mean(err[tail] ** 2)))
    if sd_pred > 0 and sd_actual > 0:
        out["slope"] = float(np.polyfit(pred, actual, 1)[0])
        out["r"] = float(np.corrcoef(pred, actual)[0, 1])

    low = actual < PRICE_BANDS["cheap"]["below"]
    high = actual > PRICE_BANDS["expensive"]["above"]
    if low.sum() > 5:
        out["low_bias"] = float(np.mean(err[low]))
    if high.sum() > 5:
        out["high_bias"] = float(np.mean(err[high]))
    return out


def forecast_report(pred, actual):
    """Full calibration + per-band detection report for a prediction/actual pair."""
    report = calibration_metrics(pred, actual)
    report["bands"] = {
        name: detection_metrics(pred, actual, **spec) for name, spec in PRICE_BANDS.items()
    }
    return report


def stored_forecast_report(stored, actuals, min_horizon_days=2.0):
    """Report on forecasts this model already published, against settled prices.

    This is the only genuinely out-of-sample gate available on a production run.
    The `build_holdout_data` holdout is not one: it excludes the forecast *runs*
    used for training but not the target *slots*, and since training keeps one run
    per day while the holdout keeps that day's other runs, 100% of holdout slots
    have had their settled price seen in training. That holdout reports sd_ratio
    ~0.94 for a model directly measured at 0.56 — which is why the defect in
    docs/MODEL_DYNAMIC_RANGE.md survived for months.

    `stored` needs columns `date_time`, `day_ahead` (the published prediction) and
    `created_at`; `actuals` is a `day_ahead`-indexed-by-datetime frame of settled
    prices. Slots closer than `min_horizon_days` are excluded because the pipeline
    blends GB60 actuals into them, so they measure the blend and not the model.
    """
    if stored is None or len(stored) == 0:
        return None
    merged = stored.merge(actuals, left_on="date_time", right_index=True, suffixes=("_pred", "_actual"))
    if merged.empty:
        return None
    horizon = (merged["date_time"] - merged["created_at"]).dt.total_seconds() / 86400.0
    merged = merged[horizon >= min_horizon_days]
    if len(merged) < 50:
        return None
    report = forecast_report(merged["day_ahead_pred"], merged["day_ahead_actual"])
    report["min_horizon_days"] = min_horizon_days
    return report


def format_report(report, label=""):
    """Render a report as a few aligned log lines."""

    def _f(value, spec=".3f"):
        return "n/a" if value is None or not np.isfinite(value) else format(value, spec)

    head = f"Forecast quality{' ' + label if label else ''} (n={report['n']}):"
    lines = [
        head,
        f"  dispersion  sd_ratio={_f(report['sd_ratio'])}  slope={_f(report['slope'])}  "
        f"r={_f(report['r'])}",
        f"  error       rmse={_f(report['rmse'], '.2f')}  mae={_f(report['mae'], '.2f')}  "
        f"tail_rmse={_f(report['tail_rmse'], '.2f')}",
        f"  regime bias low={_f(report['low_bias'], '+.2f')}  high={_f(report['high_bias'], '+.2f')}",
    ]
    for name, band in report.get("bands", {}).items():
        lines.append(
            f"  {name:<10s} recall={_f(band['recall'])}  precision={_f(band['precision'])}  "
            f"f1={_f(band['f1'])}  (n_actual={band['n_actual']}, n_flagged={band['n_flagged']})"
        )
    return "\n".join(lines)
