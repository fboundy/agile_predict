from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from pathlib import Path
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import KFold, train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from sklearn.ensemble import ExtraTreesRegressor
from django.core.cache import cache  # Or store in the database if needed
from django.db import close_old_connections, transaction

import numpy as np
import os
import logging
import time

from django.core.management.base import BaseCommand, CommandError
from ...forecast_features import (
    build_forecast_frame,
    build_holdout_data,
    build_training_data,
    EXPERIMENT_FEATURE_SETS,
    FEATURE_SETS,
    latest_prediction_features,
    add_latest_forecast_features,
    resolve_feature_columns,
    select_daily_training_forecasts,
)
from ...external_forecasts import download_daily_external_forecasts
from ...models import AgileData, ForecastData, Forecasts, History, PlotImage, PriceHistory, UpdateJob

from config.utils import *
from config.settings import GLOBAL_SETTINGS

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50
MIN_HIST = 7
MIN_FORECAST_ROWS = 200  # ~4 days; fewer rows means upstream APIs have degraded
MAX_HIST = 28
MAX_TEST_X = 10000
EXTRA_TREES_REGRESSOR_PARAMS = {
    "n_estimators": 700,
    "min_samples_leaf": 4,
    "max_features": 1.0,
    "random_state": 42,
    "n_jobs": 1,
}
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "random_seed": 42,
    "verbose": 0,
}
LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 5,
    "num_leaves": 31,
    "min_child_samples": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 3,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.environ.get("UPDATE_LOG_FILE", os.path.join(log_dir, "update.log"))

logger = logging.getLogger("prices.update")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def configure_update_logger():
    logger.handlers.clear()
    current_log_file = os.environ.get("UPDATE_LOG_FILE", log_file)
    os.makedirs(os.path.dirname(current_log_file), exist_ok=True)

    file_handler = logging.FileHandler(current_log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if os.environ.get("UPDATE_LOG_TO_CONSOLE", "1").lower() not in {"0", "false", "no", "off"}:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


configure_update_logger()


def published_plot_name(plot_path):
    try:
        return Path(plot_path).relative_to(Path("plots")).as_posix()
    except ValueError:
        return Path(plot_path).as_posix()


def clear_published_plots(prefix):
    close_old_connections()
    deleted_count, _ = PlotImage.objects.filter(filename__startswith=prefix).delete()
    cache.delete("stats_view_context")
    logger.info("Cleared %s published plot(s) with prefix %s", deleted_count, prefix)


def publish_plot(plot_path):
    close_old_connections()
    plot_path = Path(plot_path)
    filename = published_plot_name(plot_path)
    PlotImage.objects.update_or_create(
        filename=filename,
        defaults={
            "content": plot_path.read_bytes(),
            "content_type": "image/png",
        },
    )
    cache.delete("stats_view_context")
    logger.info("Published plot image %s", filename)


def refresh_db_connection(label):
    logger.info(f"Refreshing database connection: {label}")
    close_old_connections()


def lighten_cmap(cmap_name="viridis", amount=0.5):
    base = cm.get_cmap(cmap_name)
    cdict = base._segmentdata if hasattr(base, "_segmentdata") else None
    return mcolors.LinearSegmentedColormap.from_list(
        f"{cmap_name}_light",
        [
            (mcolors.to_rgba(c, alpha=1)[:3] + np.array([amount] * 3)) / (1 + amount)
            for c in base(np.linspace(0, 1, 256))
        ],
    )


def fit_day_ahead_ensemble(train_X, train_y, sample_weights):
    cat = CatBoostRegressor(**CATBOOST_PARAMS)
    cat.fit(train_X, train_y, sample_weight=sample_weights)

    lgbm = LGBMRegressor(**LGBM_PARAMS)
    lgbm.fit(train_X, train_y, sample_weight=sample_weights)

    et = ExtraTreesRegressor(**EXTRA_TREES_REGRESSOR_PARAMS)
    et.fit(train_X, train_y, sample_weight=sample_weights)

    logger.info("Fitted ensemble (CatBoost + LightGBM + ExtraTrees)")
    return [cat, lgbm, et]


def predict_day_ahead_ensemble(models, features):
    preds = np.column_stack([m.predict(features) for m in models])
    return preds.mean(axis=1)


def cross_val_ensemble_rmse(train_X, train_y, sample_weights, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=False)
    scores = []
    for tr_idx, val_idx in kf.split(train_X):
        fold_models = fit_day_ahead_ensemble(
            train_X.iloc[tr_idx], train_y.iloc[tr_idx], sample_weights.iloc[tr_idx]
        )
        preds = predict_day_ahead_ensemble(fold_models, train_X.iloc[val_idx])
        rmse = np.sqrt(np.mean((train_y.iloc[val_idx].values - preds) ** 2))
        scores.append(rmse)
        logger.info("Ensemble CV fold RMSE=%.3f", rmse)
    return np.array(scores)


def compute_horizon_quantiles(results, bins=(6, 12, 24, 36, 48)):
    """
    Compute empirical p10/p90 of (actual - predicted) residuals, binned by forecast horizon.
    Returns dict mapping upper-edge hours -> (p10_offset, p90_offset).
    """
    residuals = results["day_ahead"] - results["pred"]
    hours = results["dt"] * 24
    global_p10 = float(np.percentile(residuals, 10))
    global_p90 = float(np.percentile(residuals, 90))
    horizon_q = {}
    prev = 0
    for upper in list(bins) + [np.inf]:
        mask = (hours > prev) & (hours <= upper)
        if mask.sum() >= 5:
            horizon_q[upper] = (float(np.percentile(residuals[mask], 10)), float(np.percentile(residuals[mask], 90)))
        else:
            horizon_q[upper] = (global_p10, global_p90)
        prev = upper
    return horizon_q


_EXP_CB_PARAMS   = dict(iterations=150, learning_rate=0.05, depth=6, l2_leaf_reg=3, random_seed=42, verbose=0)
_EXP_LGBM_PARAMS = dict(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1, n_jobs=1)
_EXP_ET_PARAMS   = dict(n_estimators=200, min_samples_leaf=4, max_features=1.0, random_state=42, n_jobs=1)

# Lead-time weights: first 3 days → 3×, first 7 days → 2×, beyond → 1×
def _horizon_weights(dt_series):
    return np.where(dt_series < 3, 3.0, np.where(dt_series < 7, 2.0, 1.0))


def run_feature_experiment(df, ff_all, prices, _logger=None):
    """
    Walk-forward cross-validation across EXPERIMENT_FEATURE_SETS.
    Scores each set on weighted MAE + weighted RMSE (equal weight), where
    lead time < 3 days is weighted 3×, < 7 days 2×, else 1×.

    Returns (winning_set_name: str, results: dict[name -> {score, wmae, wrmse}])
    """
    log = _logger.info if _logger else print

    TRAIN_DAYS = 21
    TEST_DAYS  = 3
    N_FOLDS    = 5
    MIN_TRAIN  = 40

    ff_dates = ff_all.copy()
    ff_dates["_date"] = pd.to_datetime(ff_dates["name"]).dt.tz_localize("GB").dt.normalize()
    ff_dates = ff_dates.sort_values("_date").reset_index(drop=False)   # keeps 'id' col
    total_days = len(ff_dates)

    if total_days < TRAIN_DAYS + TEST_DAYS:
        log("Feature experiment: not enough forecast history, skipping")
        return "generation", {}

    log(f"Feature experiment: {total_days} daily forecasts, {N_FOLDS} folds "
        f"({TRAIN_DAYS}d train / {TEST_DAYS}d test each)")

    results = {}
    for set_name, candidate_features in EXPERIMENT_FEATURE_SETS.items():
        fold_scores = []

        for fold in range(N_FOLDS):
            end_idx   = total_days - fold * TEST_DAYS
            start_idx = max(0, end_idx - TRAIN_DAYS - TEST_DAYS)
            if end_idx - TEST_DAYS - start_idx < 2:
                continue

            train_ids = set(ff_dates.iloc[start_idx : end_idx - TEST_DAYS]["id"])
            test_ids  = set(ff_dates.iloc[end_idx - TEST_DAYS : end_idx]["id"])

            train_df = df[df["forecast_id"].isin(train_ids)].copy()
            test_df  = df[df["forecast_id"].isin(test_ids)].copy()

            # Production window filter
            train_df = train_df[
                (train_df.index >= train_df["ag_start"]) & (train_df.index < train_df["ag_end"])
            ]
            test_df = test_df[test_df.index > test_df["ag_start"]]

            # Merge actuals
            train_df = train_df.join(prices[["day_ahead"]], how="inner").dropna(subset=["day_ahead"])
            test_df  = test_df.join(prices[["day_ahead"]], how="inner").dropna(subset=["day_ahead"])

            if len(train_df) < MIN_TRAIN or len(test_df) < 10:
                continue

            # Drop features missing from df or entirely null in training
            avail = [f for f in candidate_features if f in df.columns
                     and train_df[f].notna().any()]
            if not avail:
                continue

            train_X = train_df[avail].copy()
            train_y = train_df["day_ahead"]
            test_X  = test_df[avail].copy()
            test_y  = test_df["day_ahead"]
            dt_vals = test_df["dt"].values

            col_medians = train_X.median()
            train_Xf = train_X.fillna(col_medians)
            test_Xf  = test_X.fillna(col_medians)

            preds = []
            try:
                cb = CatBoostRegressor(**_EXP_CB_PARAMS)
                cb.fit(train_X, train_y)
                preds.append(cb.predict(test_X))
            except Exception:
                pass
            try:
                lgbm = LGBMRegressor(**_EXP_LGBM_PARAMS)
                lgbm.fit(train_Xf, train_y)
                preds.append(lgbm.predict(test_Xf))
            except Exception:
                pass
            try:
                et = ExtraTreesRegressor(**_EXP_ET_PARAMS)
                et.fit(train_Xf, train_y)
                preds.append(et.predict(test_Xf))
            except Exception:
                pass

            if not preds:
                continue

            ensemble = np.mean(preds, axis=0)
            residuals = ensemble - np.array(test_y)
            weights = _horizon_weights(dt_vals)
            wmae  = float(np.average(np.abs(residuals), weights=weights))
            wrmse = float(np.sqrt(np.average(residuals ** 2, weights=weights)))
            fold_scores.append({"wmae": wmae, "wrmse": wrmse})

        if not fold_scores:
            continue

        mean_wmae  = float(np.mean([s["wmae"]  for s in fold_scores]))
        mean_wrmse = float(np.mean([s["wrmse"] for s in fold_scores]))
        score = 0.5 * mean_wmae + 0.5 * mean_wrmse
        results[set_name] = {"score": round(score, 4), "wmae": round(mean_wmae, 4), "wrmse": round(mean_wrmse, 4)}
        log(f"  {set_name:<20s}  score={score:.3f}  wmae={mean_wmae:.3f}  wrmse={mean_wrmse:.3f}  ({len(fold_scores)} folds)")

    if not results:
        log("Feature experiment: no results, defaulting to 'generation'")
        return "generation", {}

    winner = min(results, key=lambda k: results[k]["score"])
    log(f"Feature experiment winner: '{winner}'  score={results[winner]['score']:.3f}")
    return winner, results


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
            "--min_fd",
        )

        parser.add_argument(
            "--min_ad",
        )

        parser.add_argument(
            "--max_days",
        )

        parser.add_argument(
            "--no_day_of_week",
            action="store_true",
        )

        parser.add_argument(
            "--feature_set",
            choices=sorted(FEATURE_SETS),
            default=None,
            help="Named feature set to use for model training (overrides stored optimal set).",
        )

        parser.add_argument(
            "--features",
            help="Comma-separated feature columns to use instead of a named feature set.",
        )

        parser.add_argument(
            "--drop_feature",
            action="append",
            default=[],
            help="Feature column to remove from the selected feature set. May be supplied multiple times.",
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

        parser.add_argument(
            "--skip_kde_plot",
            action="store_true",
        )

    def handle(self, *args, **options):
        configure_update_logger()
        # Setup logging

        debug = options.get("debug", False)

        min_fd = int(options.get("min_fd", 600) or 600)
        min_ad = int(options.get("min_ad", 1500) or 1500)
        max_days = int(options.get("max_days", 60) or 60)

        no_ranges = options.get("no_ranges", False)
        skip_kde_plot = options.get("skip_kde_plot", False)

        # Load last experiment result; decide whether a new one is due
        _optimal_fs = "generation"
        _last_exp_date = None
        try:
            _exp_job = (
                UpdateJob.objects.filter(job_type=UpdateJob.JOB_UPDATE)
                .exclude(options__feature_experiment=None)
                .order_by("-requested_at")
                .first()
            )
            if _exp_job:
                _exp_meta = _exp_job.options["feature_experiment"]
                _optimal_fs = _exp_meta.get("feature_set", "generation")
                _last_exp_date = pd.Timestamp(_exp_meta["date"], tz="UTC")
        except Exception:
            pass

        _experiment_due = (
            _last_exp_date is None
            or (pd.Timestamp.now(tz="UTC") - _last_exp_date).days >= 14
        )
        if _experiment_due:
            logger.info("Feature experiment is due (last run: %s)", _last_exp_date or "never")
        else:
            logger.info("Feature experiment not due (last run: %s, optimal set: %s)", _last_exp_date, _optimal_fs)

        # CLI --feature_set overrides stored optimum; no CLI arg → use stored optimum
        _cli_fs = options.get("feature_set")
        features = resolve_feature_columns(
            feature_set=_cli_fs or _optimal_fs,
            explicit_features=options.get("features"),
            drop_features=options.get("drop_feature", []),
            no_day_of_week=options.get("no_day_of_week", False),
        )
        logger.info("Using model features: %s", ", ".join(features))
        download_daily_external_forecasts()

        drop_last = int(options.get("drop_last", 0) or 0)

        if options.get("ignore_forecast", []) is None:
            ignore_forecast = []
        else:
            ignore_forecast = [int(x) for x in options.get("ignore_forecast", [])]

        # Clean any invalid forecasts
        if debug:
            logger.info(f"Max days: {max_days}")

            logger.info(f"  ID  |       Name       |  #FD  |   #AD   | Days |")
            logger.info(f"------+------------------+-------+---------+------+")
        keep = []
        for f in Forecasts.objects.all().order_by("-created_at"):
            fd = ForecastData.objects.filter(forecast=f)
            ad = AgileData.objects.filter(forecast=f)
            dt = pd.to_datetime(f.name).tz_localize("GB")
            days = (pd.Timestamp.now(tz="GB") - dt).days
            if fd.count() < min_fd or ad.count() < min_ad:
                fail = " <- Fail"
            else:
                fail = " <- Manual"
                if days < max_days * 2:
                    for hour in [6, 10, 11, 16, 22]:
                        if f"{hour:02d}:15" in f.name:
                            keep.append(f.id)
                            fail = ""
                else:
                    fail = "<- Old"
            if debug:
                logger.info(f"{f.id:5d} | {f.name} | {fd.count():5d} | {ad.count():7d} | {days:4d} | {fail}")

        forecasts_to_delete = Forecasts.objects.exclude(id__in=keep)
        if debug:
            logger.info(f"\nDeleting ({forecasts_to_delete})\n")
        forecasts_to_delete.delete()

        prices, start = model_to_df(PriceHistory)

        if debug:
            logger.info("Getting Historic Prices")
            logger.info(f"Prices\n{prices}")

        agile = get_agile(start=start)
        day_ahead = day_ahead_to_agile(agile, reverse=True)

        new_prices = pd.concat([day_ahead, agile], axis=1)
        if len(prices) > 0:
            new_prices = new_prices[new_prices.index > prices.index[-1]]

        if debug:
            logger.info(f"New Prices\n{new_prices}")

        if len(new_prices) > 0:
            logger.info(new_prices)
            refresh_db_connection("before writing new price history")
            df_to_Model(new_prices, PriceHistory)
            prices = pd.concat([prices, new_prices]).sort_index()

        agile_end = prices.index[-1]
        gb60 = get_gb60()

        if debug:
            logger.info(f"GB60:\n{gb60}")

        gb60 = gb60.resample("30min").ffill().loc[agile_end + pd.Timedelta("30min") :]

        if len(gb60) > 0:
            gb60 = gb60.reindex(
                pd.date_range(gb60.index[0], gb60.index[-1] + pd.Timedelta("30min"), freq="30min")
            ).ffill()
            gb60 = pd.concat([gb60, day_ahead_to_agile(gb60)], axis=1).set_axis(["day_ahead", "agile"], axis=1)
            prices = pd.concat([prices, gb60]).sort_index()

        if debug:
            logger.info(f"Merged prices:\n{prices}")

        if drop_last > 0:
            logger.info(f"drop_last: {drop_last}")
            logger.info(f"len: {len(prices)} last:{prices.index[-1]}")
            prices = prices.iloc[:-drop_last]
            logger.info(f"len: {len(prices)} last:{prices.index[-1]}")

        new_name = pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M")
        if new_name not in [f.name for f in Forecasts.objects.all()]:
            base_forecasts = Forecasts.objects.exclude(id__in=ignore_forecast).order_by("-created_at")
            last_forecasts = {
                forecast.created_at.date(): forecast.id for forecast in base_forecasts.order_by("created_at")
            }

            base_forecasts = base_forecasts.filter(id__in=[last_forecasts[k] for k in last_forecasts])

            if debug:
                logger.info("Getting latest Forecast")

            fc, missing_fc, source_rows, source_details = get_latest_forecast()
            refresh_db_connection("after fetching latest forecast")

            # Persist source row counts and per-source details into the running UpdateJob
            # so the UI can read traffic-light health across processes.
            api_status_data = {
                "source_rows": source_rows,
                "source_details": source_details,
                "forecast_rows": len(fc),
                "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
            }
            try:
                running_job = UpdateJob.objects.filter(
                    job_type=UpdateJob.JOB_UPDATE,
                ).order_by("-requested_at").first()
                if running_job:
                    running_job.options["api_status"] = api_status_data
                    running_job.save(update_fields=["options"])
            except Exception:
                pass
            logger.info("Upstream source rows: %s  total forecast rows: %d", source_rows, len(fc))

            if len(missing_fc) > 0:
                raise CommandError(
                    f"Upstream APIs missing columns: {', '.join(missing_fc)}. Aborting to avoid degraded forecast."
                )

            if len(fc) < MIN_FORECAST_ROWS:
                raise CommandError(
                    f"Upstream APIs returned only {len(fc)} forecast rows (minimum {MIN_FORECAST_ROWS}). "
                    "Aborting to avoid degraded forecast."
                )

            if True:
                    refresh_db_connection("before loading training data")
                    fd = pd.DataFrame(list(ForecastData.objects.exclude(forecast_id__in=ignore_forecast).values()))
                    ff = pd.DataFrame(list(Forecasts.objects.exclude(id__in=ignore_forecast).values()))

                    if len(ff) > 0:
                        logger.info(ff)
                        df, ff = build_forecast_frame(fd, ff)
                        ff_train = select_daily_training_forecasts(ff)

                        # ── Periodic feature experiment ───────────────────────
                        if _experiment_due and not _cli_fs and not options.get("features"):
                            try:
                                logger.info("Running feature set experiment…")
                                _winner, _exp_results = run_feature_experiment(df, ff_train, prices, logger)
                                # Update features for this run if the winner differs
                                if _winner != _optimal_fs:
                                    logger.info(
                                        "Feature set changing: %s → %s", _optimal_fs, _winner
                                    )
                                    features = list(EXPERIMENT_FEATURE_SETS[_winner])
                                else:
                                    logger.info("Feature set unchanged: %s", _winner)
                                # Persist result
                                _exp_payload = {
                                    "date": pd.Timestamp.now(tz="UTC").isoformat(),
                                    "feature_set": _winner,
                                    "results": _exp_results,
                                }
                                _save_job = UpdateJob.objects.filter(
                                    job_type=UpdateJob.JOB_UPDATE,
                                ).order_by("-requested_at").first()
                                if _save_job:
                                    _save_job.options["feature_experiment"] = _exp_payload
                                    _save_job.save(update_fields=["options"])
                                _experiment_due = False
                            except Exception:
                                logger.exception("Feature experiment failed; keeping existing feature set")
                        # ─────────────────────────────────────────────────────

                        if debug:
                            logger.info(f"Forecasts Database:\n{ff.to_string()}")

                        # Only use the forecasts closest to 16:15 for training
                        train_X, train_y = build_training_data(df, ff_train, prices, features, max_days)
                        if debug:
                            logger.info(f"train_X:\n{train_X}")
                        sample_weights = ((np.log10((train_y - train_y.mean()).abs() + 10) * 5) - 4).round(0)

                        logger.info(
                            "Computing ensemble cross-validation score "
                            f"({len(train_X)} rows, {len(train_X.columns)} features)"
                        )
                        scores = cross_val_ensemble_rmse(train_X, train_y, sample_weights)
                        logger.info(f"Ensemble cross-val RMSE: {scores}")
                        refresh_db_connection("after cross-validation")

                        logger.info(
                            "Fitting final ensemble model "
                            f"({len(train_X)} rows, {len(train_X.columns)} features)"
                        )
                        ensemble_models = fit_day_ahead_ensemble(train_X, train_y, sample_weights)
                        logger.info("Finished fitting ensemble model")
                        refresh_db_connection("after fitting ensemble model")

                        # Drop the training data set
                        logger.info("Preparing holdout/test dataset")
                        test_X, test_y = build_holdout_data(df, ff_train, prices, max_days)

                        if len(test_X) > MAX_TEST_X:
                            logger.info(f"Sampling test dataset from {len(test_X)} rows to {MAX_TEST_X} rows")
                            _, test_X, _, _ = train_test_split(test_X, test_y, test_size=MAX_TEST_X)

                        if debug:
                            logger.info(f"len(ff)      : {len(ff)}")
                            logger.info(f"len(ff_train): {len(ff_train)}")
                            logger.info(f"len(train_X) : {len(train_X)}")
                            logger.info(f"len(test_X)  : {len(test_X)}")

                            logger.info(f"Earliest ff   : {ff.index.min()}")
                            logger.info(f"Latest ff     : {ff.index.max()}")
                            logger.info(f"Earliest ff_t : {ff_train.index.min()}")
                            logger.info(f"Latest ff_t   : {ff_train.index.max()}")

                            logger.info("train_cols:")
                            for col in train_X.columns:
                                logger.info(
                                    f"  {col:16s}:  {train_X[col].min():10.2f} {train_X[col].mean():10.2f} {train_X[col].max():10.2f}"
                                )

                            logger.info(f"test_X:\n{test_X}")

                        factor = GLOBAL_SETTINGS["REGIONS"]["X"]["factors"][0]

                        logger.info(f"Predicting holdout/test dataset ({len(test_X)} rows)")
                        results = test_X[["dt", "day_ahead"]].copy()
                        results["pred"] = predict_day_ahead_ensemble(ensemble_models, test_X[features])
                        logger.info("Finished holdout/test predictions")

                        # Add required columns before plotting
                        results["forecast_created"] = test_X["created_at"]
                        results["target_time"] = test_X.index
                        results["next_agile"] = (test_X.index >= test_X["ag_start"]) & (
                            test_X.index < test_X["ag_end"]
                        )
                        results["error"] = (results["day_ahead"] - results["pred"]) * factor

                        def save_plot(fig, name):
                            plot_path = os.path.join(PLOT_DIR, f"{name}.png")
                            fig.savefig(plot_path, bbox_inches="tight")
                            plt.close(fig)
                            publish_plot(plot_path)

                        PLOT_DIR = Path(os.path.join("plots", "trends"))
                        logger.info(f"Writing trend plot to {PLOT_DIR}")
                        PLOT_DIR.mkdir(parents=True, exist_ok=True)
                        for f in PLOT_DIR.glob("*.png"):
                            f.unlink()
                        clear_published_plots("trends/")

                        fig, ax = plt.subplots(figsize=(16, 6))
                        ff = pd.concat(
                            [
                                ff,
                                pd.DataFrame(
                                    index=[ff.index[-1] + 1],
                                    data={
                                        "created_at": [pd.Timestamp(new_name, tz="GB")],
                                        "mean": [np.mean(scores)],
                                        "stdev": [np.std(scores)],
                                    },
                                ),
                            ]
                        )

                        ax.plot(ff["created_at"], ff["mean"] * factor, lw=2, color="black", marker="o")
                        ax.fill_between(
                            ff["created_at"],
                            (ff["mean"] - ff["stdev"]) * factor,
                            (ff["mean"] + ff["stdev"]) * factor,
                            color="yellow",
                            alpha=0.3,
                            label="±1 Stdev",
                        )

                        ax.set_ylabel("Predicted Agile Price RMSE [p/kWh]")
                        ax.set_xlabel("Forecast Date/Time")
                        ax.set_ylim(0)
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
                        fig.autofmt_xdate()  # rotates and aligns labels
                        save_plot(fig, "trend")
                        logger.info("Saved trend plot")

                        # Directory to save plots
                        PLOT_DIR = Path(os.path.join("plots", "stats_plots"))
                        logger.info(f"Writing stats plots to {PLOT_DIR}")
                        PLOT_DIR.mkdir(parents=True, exist_ok=True)

                        # Clean old files (optional)
                        for f in PLOT_DIR.glob("*.png"):
                            f.unlink()
                        clear_published_plots("stats_plots/")

                        # 1. Prediction vs Actual over Time
                        fig, ax = plt.subplots(figsize=(16, 6))

                        subset = results[results["next_agile"]].sort_values("target_time")
                        ax.plot(subset["target_time"], subset["day_ahead"], label="Actual", color="black")
                        ax.plot(
                            subset["target_time"],
                            subset["pred"],
                            label="Trained Model Prediction",
                            alpha=0.4,
                            markersize=2.5,
                            color="red",
                            lw=0,
                            marker="o",
                        )

                        subset = results[~results["next_agile"]].sort_values("target_time")
                        sc = ax.scatter(
                            x=subset["target_time"],
                            y=subset["pred"],
                            label="Predicted",
                            alpha=0.4,
                            c=subset["dt"],
                            lw=0,
                            marker="o",
                            cmap="viridis",
                        )
                        cbar = fig.colorbar(sc, ax=ax)
                        cbar.set_label("Days Ahead (dt)")

                        # Format datetime axis
                        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
                        fig.autofmt_xdate()  # rotates and aligns labels

                        ax.set_title("Training Dataset - Actual vs Predicted")
                        ax.set_ylabel("£/MWh")
                        ax.legend()
                        save_plot(fig, "1_actual_vs_predicted_over_time")
                        logger.info("Saved stats plot 1/5: actual vs predicted over time")

                        # 2. Prediction vs Actual Scatter
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sc = ax.scatter(
                            results["day_ahead"], results["pred"], alpha=0.2, c=results["dt"], cmap="plasma"
                        )
                        cbar = fig.colorbar(sc, ax=ax)
                        cbar.set_label("Days Ahead (dt)")
                        ax.plot(
                            [results["day_ahead"].min(), results["day_ahead"].max()],
                            [results["day_ahead"].min(), results["day_ahead"].max()],
                            "--",
                            color="gray",
                        )
                        ax.set_xlabel("Actual Day-Ahead Price [£/MWh]")
                        ax.set_ylabel("Predicted Price [£/MWh]")
                        ax.set_title("Prediction vs Actual")
                        save_plot(fig, "2_scatters")
                        logger.info("Saved stats plot 2/5: prediction vs actual scatter")

                        # 3. Residuals
                        fig, ax = plt.subplots(figsize=(8, 6))
                        residuals = (results["day_ahead"] - results["pred"]) * factor
                        sns.histplot(residuals, bins=50, kde=True, ax=ax)
                        ax.set_title("Residuals Distribution")
                        ax.set_xlabel("Error (Actual - Predicted) [p/kWh]")
                        save_plot(fig, "3_residuals")
                        logger.info("Saved stats plot 3/5: residuals distribution")

                        if skip_kde_plot:
                            logger.info("Skipped stats plot 4/5: forecast error by horizon KDE")
                        else:
                            # 4. Forecast Error by Horizon
                            logger.info("Generating stats plot 4/5: forecast error by horizon KDE")
                            fig, ax = plt.subplots(figsize=(8, 6))
                            kde = sns.kdeplot(
                                data=results,
                                x="dt",
                                y="error",
                                fill=True,
                                cmap="Oranges",
                                levels=10,
                                ax=ax,
                            )

                            # Add a colorbar
                            # cbar = plt.colorbar(kde.collections[0], ax=ax)
                            # cbar.set_label("Density")
                            # sns.scatterplot(
                            #     data=results,
                            #     x="dt",
                            #     y=residuals,
                            #     alpha=0.3,
                            #     ax=ax,
                            #     color="grey",
                            #     linewidth=0,
                            # )
                            ax.set_title("2D KDE: Forecast Error by Horizon")
                            ax.set_xlabel("Days Ahead (dt)")
                            ax.set_ylabel("Error (Actual - Predicted) [p/kWh]")
                            save_plot(fig, "4_kde_error_by_horizon")
                            logger.info("Saved stats plot 4/5: forecast error by horizon")

                        # 5. Feature Importance (ensemble average of normalised importances)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        cat_imp = ensemble_models[0].get_feature_importance().astype(float)
                        lgbm_imp = ensemble_models[1].feature_importances_.astype(float)
                        et_imp = ensemble_models[2].feature_importances_.astype(float)
                        avg_imp = (cat_imp / cat_imp.sum() + lgbm_imp / lgbm_imp.sum() + et_imp / et_imp.sum()) / 3
                        pd.Series(avg_imp, index=features).sort_values().plot.barh(ax=ax)
                        ax.set_title("Ensemble Feature Importance (Average)")
                        ax.set_xlabel("Relative Importance")
                        save_plot(fig, "5_feature_importance")
                        logger.info("Saved stats plot 5/5: feature importance")

                        # Persist feature importances to the running UpdateJob so the v2 stats
                        # page can render them as a Plotly chart without filesystem access.
                        try:
                            _fi_dict = {
                                str(f): round(float(v), 6)
                                for f, v in zip(features, avg_imp.tolist())
                            }
                            _fi_job = UpdateJob.objects.filter(
                                job_type=UpdateJob.JOB_UPDATE,
                            ).order_by("-requested_at").first()
                            if _fi_job:
                                _fi_job.options["feature_importance"] = _fi_dict
                                _fi_job.save(update_fields=["options"])
                        except Exception:
                            pass

                        # fig, ax = plt.subplots(figsize=(8, 6))
                        # bins = [0, 1, 2, 3, 5, 10, 15]
                        # labels = [f"{i}-{j}" for i, j in zip(bins[:-1], bins[1:])]
                        # results["horizon_bucket"] = pd.cut(results["dt"], bins=bins, labels=labels, right=True)
                        # ax = sns.violinplot(data=results, x="horizon_bucket", y="error")
                        # ax.set_xlabel("Days Ahead (dt)")
                        # ax.set_ylabel("Error (Actual - Predicted) [£/MWh]")
                        # ax.set_title("Error Distribution by Time Horion Bin")
                        # ax.legend()
                        # save_plot(fig, "6_binned_error_v_time")

                    fc = add_latest_forecast_features(fc)
                    if len(ff) > 0:
                        logger.info(f"Predicting latest forecast ({len(fc)} rows)")
                        prediction_features = latest_prediction_features(fc, train_X.columns)
                        fc["day_ahead"] = predict_day_ahead_ensemble(ensemble_models, prediction_features)
                        logger.info("Finished latest forecast prediction")

                        fc["day_ahead_classified"] = np.nan
                        fc["day_ahead_extra_trees"] = np.nan
                        fc["plunge_probability"] = np.nan

                        if (len(test_X) > 10) and (not no_ranges):
                            interval_started = time.monotonic()

                            # Empirical horizon quantiles from holdout residuals
                            horizon_q = compute_horizon_quantiles(results)
                            logger.info(
                                "Horizon quantiles (p10/p90 by hours-ahead): %s",
                                {
                                    ("inf" if k == np.inf else int(k)): (f"{v[0]:.2f}", f"{v[1]:.2f}")
                                    for k, v in horizon_q.items()
                                },
                            )
                            sorted_keys = sorted(horizon_q)
                            fc_hours = fc["dt"] * 24
                            fc["day_ahead_low"] = fc["day_ahead"] + fc_hours.apply(
                                lambda h: next((horizon_q[k] for k in sorted_keys if h <= k), horizon_q[sorted_keys[-1]])[0]
                            )
                            fc["day_ahead_high"] = fc["day_ahead"] + fc_hours.apply(
                                lambda h: next((horizon_q[k] for k in sorted_keys if h <= k), horizon_q[sorted_keys[-1]])[1]
                            )

                            # Ensemble weather for day-adaptive intervals
                            weather_members = get_weather_ensemble()
                            if weather_members:
                                logger.info("Running model on %d weather ensemble members", len(weather_members))
                                member_preds = []
                                for member in weather_members:
                                    member_fc = fc.copy()
                                    for col in ["temp_2m", "wind_10m", "rad"]:
                                        if col in member.columns and col in member_fc.columns:
                                            member_fc[col] = (
                                                member[col].reindex(member_fc.index).ffill().bfill()
                                            )
                                    member_preds.append(
                                        predict_day_ahead_ensemble(
                                            ensemble_models,
                                            latest_prediction_features(member_fc, train_X.columns),
                                        )
                                    )
                                member_arr = np.column_stack(member_preds)
                                ens_low = np.percentile(member_arr, 10, axis=1)
                                ens_high = np.percentile(member_arr, 90, axis=1)
                                fc["day_ahead_low"] = np.minimum(fc["day_ahead_low"].values, ens_low)
                                fc["day_ahead_high"] = np.maximum(fc["day_ahead_high"].values, ens_high)
                                logger.info("Applied ensemble weather intervals members=%d", len(weather_members))
                            else:
                                logger.info("Ensemble weather unavailable, using empirical quantiles only")

                            for case in ["low", "high"]:
                                fc[f"day_ahead_{case}"] = (
                                    fc[f"day_ahead_{case}"].rolling(3, center=True).mean().bfill().ffill()
                                )
                            fc["day_ahead_low"] = fc[["day_ahead", "day_ahead_low"]].min(axis=1)
                            fc["day_ahead_high"] = fc[["day_ahead", "day_ahead_high"]].max(axis=1)
                            logger.info(
                                "Finished forecast intervals elapsed_seconds=%.2f",
                                time.monotonic() - interval_started,
                            )

                        else:
                            logger.info("Using fallback +/-10% forecast ranges")
                            fc["day_ahead_low"] = fc["day_ahead"] * 0.9
                            fc["day_ahead_high"] = fc["day_ahead"] * 1.1

                    else:
                        fc["day_ahead"] = None
                        fc["day_ahead_classified"] = None
                        fc["day_ahead_extra_trees"] = None
                        fc["plunge_probability"] = None
                        fc["day_ahead_low"] = None
                        fc["day_ahead_high"] = None

                    if debug:
                        logger.info(f"Forecast from {fc.index[0]} tp {fc.index[-1]}")
                        logger.info(f"Agile to      {agile_end}")
                        if len(gb60) > 0:
                            logger.info(f"GB60 to       {prices.index[-1]}")

                        logger.info(f"Forecast\n{fc}")

                    sfs = [
                        pd.DataFrame(
                            index=pd.date_range(fc.index[0], agile_end, freq="30min"), data={"mult": 0, "shift": 1}
                        )
                    ]

                    if len(gb60) > 0:
                        sfs.append(
                            pd.DataFrame(
                                index=pd.date_range(gb60.index[0], prices.index[-1], freq="30min"),
                                data={"mult": 0, "shift": 5},
                            )
                        )
                        sfs.append(
                            pd.DataFrame(
                                index=fc.index.difference(sfs[0].index.union(sfs[1].index)),
                                data={"mult": 1, "shift": 0},
                            )
                        )
                    else:
                        sfs.append(pd.DataFrame(index=fc.index.difference(sfs[0].index), data={"mult": 1, "shift": 0}))

                    fc = fc.astype(float)
                    scale_factors = pd.concat(sfs)

                    if debug:
                        for i, sf in enumerate(sfs):
                            if len(sf.index) > 0:
                                logger.info(f"idx{i}: {sf.index[0]}:{sf.index[-1]}\n{sf}")
                        logger.info(f"Scale factors\n{scale_factors}")

                    scale_factors = pd.concat([scale_factors, prices.reindex(scale_factors.index).fillna(0)], axis=1)
                    logger.info(f"Applying scale factors ({len(scale_factors)} rows)")

                    if debug:
                        logger.info(f"Scale Factors:\n{scale_factors}")

                    fc["day_ahead"] = fc["day_ahead"] * scale_factors["mult"] + scale_factors["day_ahead"] * (
                        1 - scale_factors["mult"]
                    )
                    fc["day_ahead_low"] = (
                        fc["day_ahead_low"] * scale_factors["mult"]
                        + scale_factors["day_ahead"] * (1 - scale_factors["mult"])
                        - scale_factors["shift"]
                    )
                    fc["day_ahead_high"] = (
                        fc["day_ahead_high"] * scale_factors["mult"]
                        + scale_factors["day_ahead"] * (1 - scale_factors["mult"])
                        + scale_factors["shift"]
                    )

                    if debug:
                        logger.info(
                            pd.concat(
                                [
                                    scale_factors,
                                    fc[["day_ahead", "day_ahead_low", "day_ahead_high"]],
                                ],
                                axis=1,
                            )
                        )

                    agile_regions = [
                        region for region in regions if not GLOBAL_SETTINGS["REGIONS"][region].get("raw_day_ahead")
                    ]
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
                            for region in agile_regions
                        ]
                    )
                    logger.info(f"Prepared agile forecast data ({len(ag)} rows across {len(agile_regions)} regions)")

                    # fc = fc[list(fd.columns)[3:]]
                    fc = fc[
                        [
                            "bm_wind",
                            "solar",
                            "emb_wind",
                            "nuclear",
                            "gas_ttf",
                            "temp_2m",
                            "wind_10m",
                            "rad",
                            "demand",
                            "day_ahead",
                            "day_ahead_classified",
                            "day_ahead_extra_trees",
                            "plunge_probability",
                            *[c for c in ("fr_nuclear", "opmr_surplus") if c in fc.columns],
                        ]
                    ]

                    if debug:
                        logger.info(f"Final forecast from {fc.index[0]} to {fc.index[-1]}")
                        logger.info(f"Forecast\n{fc}")

                    refresh_db_connection("before saving forecast rows")
                    with transaction.atomic():
                        this_forecast = Forecasts(name=new_name, mean=np.mean(scores), stdev=np.std(scores))
                        logger.info(f"Saving forecast record: {new_name}")
                        this_forecast.save()
                        fc["forecast"] = this_forecast
                        ag["forecast"] = this_forecast
                        logger.info(f"Writing ForecastData rows: {len(fc)}")
                        df_to_Model(fc, ForecastData)
                        logger.info(f"Writing AgileData rows: {len(ag)}")
                        df_to_Model(ag, AgileData)
                    logger.info(f"Finished writing forecast {this_forecast.id}: {this_forecast.name}")

        if debug:
            for f in Forecasts.objects.all().order_by("-created_at"):
                logger.info(f"{f.id:4d}: {f.name}")
        else:
            try:
                logger.info(f"\n\nAdded Forecast: {this_forecast.id:>4d}: {this_forecast.name}")
            except:
                logger.info("No forecast added")
