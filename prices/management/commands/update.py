import xgboost as xg
from pathlib import Path
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns

from django.core.cache import cache  # Or store in the database if needed
from django.db import close_old_connections

import numpy as np
import os
import logging
import time

from django.core.management.base import BaseCommand
from ...forecast_features import (
    build_forecast_frame,
    build_holdout_data,
    build_training_data,
    FEATURE_SETS,
    latest_prediction_features,
    add_latest_forecast_features,
    resolve_feature_columns,
    select_daily_training_forecasts,
)
from ...external_forecasts import download_daily_external_forecasts
from ...models import AgileData, ForecastData, Forecasts, History, PlotImage, PriceHistory

from config.utils import *
from config.settings import GLOBAL_SETTINGS

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50
MIN_HIST = 7
MAX_HIST = 28
MAX_TEST_X = 10000
PLUNGE_DAY_AHEAD_THRESHOLD = 34.1893
CLASSIFIED_MODEL_FEATURES = [
    "bm_wind",
    "emb_wind",
    "solar",
    "demand",
    "wind_10m",
    "days_ago",
    "weekend",
    "peak",
]

XGBOOST_PARAMETER_SETS = {
    "current_dart": {
        "objective": "reg:squarederror",
        "booster": "dart",
        "gamma": 0.2,
        "subsample": 1.0,
        "n_estimators": 200,
        "max_depth": 10,
        "colsample_bytree": 1,
        "n_jobs": 1,
    },
    "conservative_gbtree": {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.1,
        "reg_lambda": 3,
        "n_jobs": 1,
        "random_state": 42,
    },
    "shallow_regularized": {
        "objective": "reg:squarederror",
        "booster": "gbtree",
        "learning_rate": 0.05,
        "n_estimators": 800,
        "max_depth": 3,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0,
        "reg_lambda": 10,
        "reg_alpha": 0.1,
        "n_jobs": 1,
        "random_state": 42,
    },
    "regularized_dart": {
        "objective": "reg:squarederror",
        "booster": "dart",
        "learning_rate": 0.05,
        "n_estimators": 500,
        "max_depth": 5,
        "min_child_weight": 3,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "gamma": 0.1,
        "reg_lambda": 3,
        "rate_drop": 0.1,
        "skip_drop": 0.5,
        "n_jobs": 1,
        "random_state": 42,
    },
}
CLASSIFIED_XGBOOST_CLASSIFIER_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "booster": "gbtree",
    "learning_rate": 0.03,
    "n_estimators": 800,
    "max_depth": 2,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 0,
    "reg_lambda": 1,
    "random_state": 42,
    "n_jobs": 1,
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


def fit_classified_day_ahead_models(train_X, train_y, sample_weights):
    plunge_class = (train_y <= PLUNGE_DAY_AHEAD_THRESHOLD).astype(int)
    plunge_count = int(plunge_class.sum())
    normal_count = int(len(plunge_class) - plunge_count)
    if plunge_count == 0 or normal_count == 0:
        logger.warning(
            "Skipping classified day-ahead model because training classes are incomplete "
            "plunge_rows=%s normal_rows=%s",
            plunge_count,
            normal_count,
        )
        return None

    scale_pos_weight = normal_count / plunge_count * 2
    classifier = xg.XGBClassifier(**CLASSIFIED_XGBOOST_CLASSIFIER_PARAMS, scale_pos_weight=scale_pos_weight)
    classifier.fit(train_X, plunge_class)

    plunge_mask = plunge_class == 1
    normal_mask = ~plunge_mask.astype(bool)

    plunge_model = xg.XGBRegressor(**XGBOOST_PARAMETER_SETS["conservative_gbtree"])
    normal_model = xg.XGBRegressor(**XGBOOST_PARAMETER_SETS["conservative_gbtree"])
    plunge_model.fit(train_X.loc[plunge_mask], train_y.loc[plunge_mask], sample_weight=sample_weights.loc[plunge_mask])
    normal_model.fit(train_X.loc[normal_mask], train_y.loc[normal_mask], sample_weight=sample_weights.loc[normal_mask])

    logger.info(
        "Fitted classified day-ahead models plunge_rows=%s normal_rows=%s threshold=%.4f",
        plunge_count,
        normal_count,
        PLUNGE_DAY_AHEAD_THRESHOLD,
    )
    return classifier, plunge_model, normal_model


def predict_classified_day_ahead(models, features):
    classifier, plunge_model, normal_model = models
    plunge_probability = classifier.predict_proba(features)[:, 1]
    plunge_prediction = plunge_model.predict(features)
    normal_prediction = normal_model.predict(features)
    day_ahead_classified = plunge_probability * plunge_prediction + (1 - plunge_probability) * normal_prediction
    return day_ahead_classified, plunge_probability


def kde_quantiles(kde, dt, pred, quantiles={"low": 0.1, "mid": 0.5, "high": 0.9}, lim=(0, 150), log_every=25):
    if not isinstance(dt, list):
        dt = [dt]
    if not isinstance(pred, list):
        pred = [pred]

    results = {q: [] for q in quantiles}
    total = len(dt)
    started = time.monotonic()
    price_points = len(range(int(lim[0]), int(lim[1])))
    logger.info(
        "Starting KDE quantiles rows=%s price_points_per_row=%s quantiles=%s lim=%s",
        total,
        price_points,
        list(quantiles.keys()),
        lim,
    )
    for row_number, (dt1, pred1) in enumerate(zip(dt, pred), start=1):
        if row_number == 1 or row_number % log_every == 0 or row_number == total:
            logger.info(
                "KDE quantiles progress row=%s/%s elapsed_seconds=%.2f",
                row_number,
                total,
                time.monotonic() - started,
            )

        row_started = time.monotonic()
        x = np.array([[dt1, pred1, p] for p in range(int(lim[0]), int(lim[1]))])
        c = pd.Series(index=x[:, 2], data=np.exp(kde.score_samples(x)).cumsum())
        c /= c.iloc[-1]
        if time.monotonic() - row_started > 5:
            logger.info(
                "Slow KDE quantile row row=%s/%s dt=%.4f pred=%.4f duration_seconds=%.2f",
                row_number,
                total,
                dt1,
                pred1,
                time.monotonic() - row_started,
            )

        for q in quantiles:
            if len(c[c < quantiles[q]]) > 0:
                idx = c[c < quantiles[q]].index[-1]
                results[q] += [(quantiles[q] - c[idx]) / (c[idx + 1] - c[idx]) + idx]
            else:
                results[q] += [np.nan]
    logger.info("Finished KDE quantiles rows=%s elapsed_seconds=%.2f", total, time.monotonic() - started)
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
            default="generation",
            help="Named feature set to use for model training.",
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
            "--xgboost_params",
            choices=sorted(XGBOOST_PARAMETER_SETS),
            default="regularized_dart",
            help="Named XGBoost parameter set to use for model training.",
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

        features = resolve_feature_columns(
            feature_set=options.get("feature_set", "generation"),
            explicit_features=options.get("features"),
            drop_features=options.get("drop_feature", []),
            no_day_of_week=options.get("no_day_of_week", False),
        )
        logger.info("Using model features: %s", ", ".join(features))
        xgboost_params_name = options.get("xgboost_params", "regularized_dart")
        xgboost_params = XGBOOST_PARAMETER_SETS[xgboost_params_name]
        logger.info("Using XGBoost parameter set: %s", xgboost_params_name)
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

            fc, missing_fc = get_latest_forecast()
            refresh_db_connection("after fetching latest forecast")

            if len(missing_fc) > 0:
                logger.error(f">>> ERROR: Unable to run forecast due to missing columns: {', '.join(missing_fc)}")
            else:
                if debug:
                    logger.info(fc)

                if len(fc) > 0:
                    refresh_db_connection("before loading training data")
                    fd = pd.DataFrame(list(ForecastData.objects.exclude(forecast_id__in=ignore_forecast).values()))
                    ff = pd.DataFrame(list(Forecasts.objects.exclude(id__in=ignore_forecast).values()))

                    if len(ff) > 0:
                        logger.info(ff)
                        df, ff = build_forecast_frame(fd, ff)
                        ff_train = select_daily_training_forecasts(ff)

                        if debug:
                            logger.info(f"Forecasts Database:\n{ff.to_string()}")

                        # Only use the forecasts closest to 16:15 for training
                        train_X, train_y = build_training_data(df, ff_train, prices, features, max_days)
                        if debug:
                            logger.info(f"train_X:\n{train_X}")
                        sample_weights = ((np.log10((train_y - train_y.mean()).abs() + 10) * 5) - 4).round(0)

                        xg_model = xg.XGBRegressor(**xgboost_params)

                        scores = cross_val_score(
                            xg_model, train_X, train_y, cv=5, scoring="neg_root_mean_squared_error"
                        )

                        logger.info(f"Cross-val score: {scores}")
                        refresh_db_connection("after cross-validation")

                        logger.info(
                            "Fitting final XGBoost model "
                            f"({len(train_X)} rows, {len(train_X.columns)} features)"
                        )
                        xg_model.fit(train_X, train_y, sample_weight=sample_weights, verbose=True)
                        logger.info("Finished fitting final XGBoost model")
                        refresh_db_connection("after fitting final XGBoost model")

                        classified_models = None
                        try:
                            classified_train_X, classified_train_y = build_training_data(
                                df, ff_train, prices, CLASSIFIED_MODEL_FEATURES, max_days
                            )
                            classified_sample_weights = (
                                (np.log10((classified_train_y - classified_train_y.mean()).abs() + 10) * 5) - 4
                            ).round(0)
                            logger.info(
                                "Fitting classified day-ahead models "
                                f"({len(classified_train_X)} rows, {len(classified_train_X.columns)} features)"
                            )
                            classified_models = fit_classified_day_ahead_models(
                                classified_train_X,
                                classified_train_y,
                                classified_sample_weights,
                            )
                            refresh_db_connection("after fitting classified day-ahead models")
                        except ValueError:
                            logger.exception("Unable to fit classified day-ahead models")

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
                        results["pred"] = xg_model.predict(test_X[features])
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
                                        "mean": [-np.mean(scores)],
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

                        # 5. Feature Importance (XGBoost built-in)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        xg.plot_importance(xg_model, ax=ax, importance_type="gain", show_values=False)
                        ax.set_title("XGBoost Feature Importance (Gain)")
                        save_plot(fig, "5_feature_importance")
                        logger.info("Saved stats plot 5/5: feature importance")

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
                        fc["day_ahead"] = xg_model.predict(latest_prediction_features(fc, train_X.columns))
                        logger.info("Finished latest forecast prediction")

                        if classified_models is not None:
                            logger.info("Predicting latest forecast with classified day-ahead model")
                            classified_features = latest_prediction_features(fc, CLASSIFIED_MODEL_FEATURES)
                            fc["day_ahead_classified"], fc["plunge_probability"] = predict_classified_day_ahead(
                                classified_models, classified_features
                            )
                            logger.info("Finished classified day-ahead prediction")
                        else:
                            fc["day_ahead_classified"] = None
                            fc["plunge_probability"] = None

                        if (len(test_X) > 10) and (not no_ranges):
                            kde_fit_data = results[["dt", "pred", "day_ahead"]].to_numpy()
                            logger.info(
                                "Fitting KDE for forecast range estimates rows=%s cols=%s",
                                kde_fit_data.shape[0],
                                kde_fit_data.shape[1],
                            )
                            kde_started = time.monotonic()
                            kde = KernelDensity()
                            kde.fit(kde_fit_data)
                            logger.info("Finished KDE fit elapsed_seconds=%.2f", time.monotonic() - kde_started)

                            xlim = (
                                np.floor(results[["pred", "day_ahead"]].min(axis=1).min() / 11) * 10,
                                np.ceil(results[["pred", "day_ahead"]].max(axis=1).max() / 9) * 10,
                            )
                            logger.info(
                                "Calculating KDE forecast range quantiles forecast_rows=%s xlim=%s",
                                len(fc),
                                xlim,
                            )
                            quantile_started = time.monotonic()

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
                            logger.info(
                                "Finished KDE forecast range estimates elapsed_seconds=%.2f",
                                time.monotonic() - quantile_started,
                            )

                            for case in ["low", "high"]:
                                fc[f"day_ahead_{case}"] = (
                                    fc[f"day_ahead_{case}"].rolling(3, center=True).mean().bfill().ffill()
                                )

                            fc["day_ahead_low"] = fc[["day_ahead", "day_ahead_low"]].min(axis=1)
                            fc["day_ahead_high"] = fc[["day_ahead", "day_ahead_high"]].max(axis=1)

                        else:
                            logger.info("Using fallback +/-10% forecast ranges")
                            fc["day_ahead_low"] = fc["day_ahead"] * 0.9
                            fc["day_ahead_high"] = fc["day_ahead"] * 1.1

                    else:
                        fc["day_ahead"] = None
                        fc["day_ahead_classified"] = None
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
                    fc["day_ahead_classified"] = fc["day_ahead_classified"] * scale_factors["mult"] + scale_factors[
                        "day_ahead"
                    ] * (1 - scale_factors["mult"])
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
                            pd.concat([scale_factors, fc[["day_ahead", "day_ahead_low", "day_ahead_high"]]], axis=1)
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
                    logger.info(f"Prepared agile forecast data ({len(ag)} rows across {len(regions)} regions)")

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
                            "plunge_probability",
                        ]
                    ]

                    if debug:
                        logger.info(f"Final forecast from {fc.index[0]} to {fc.index[-1]}")
                        logger.info(f"Forecast\n{fc}")

                    refresh_db_connection("before saving forecast rows")
                    this_forecast = Forecasts(name=new_name, mean=-np.mean(scores), stdev=np.std(scores))
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
