import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand

from config.utils import day_ahead_to_agile
from prices.forecast_features import select_daily_training_forecasts
from prices.models import ForecastData, Forecasts, PriceHistory
from prices.timestral import forecast_day_ahead, load_forecaster


def first_unpriced_slot(run, prices):
    """First slot of a run that it forecast rather than copied from known prices.

    The update splices known prices into a run's day_ahead verbatim, so the first
    slot whose stored value differs from the price history is where the model took
    over (GB60 slots also differ, so they count as forecast here).
    """
    fd = pd.Series(dict(ForecastData.objects.filter(forecast=run).values_list("date_time", "day_ahead")), dtype=float)
    fd.index = pd.DatetimeIndex(fd.index).tz_convert("UTC")
    fd = fd.sort_index()
    known = (fd - prices.reindex(fd.index)).abs() <= 0.01
    unpriced = fd.index[~known.to_numpy()]
    return (unpriced[0] if len(unpriced) else fd.index[-1]), fd


class Command(BaseCommand):
    help = (
        "Backtest Timestral against stored daily runs: reconstruct each run's inputs as known "
        "at the time, forecast, and compare with the run's own ensemble forecast and the outturn."
    )

    def add_arguments(self, parser):
        parser.add_argument("--model", default=settings.TIMESTRAL_MODEL)
        parser.add_argument("--context-days", type=int, default=settings.TIMESTRAL_CONTEXT_DAYS)
        parser.add_argument("--start", help="Only runs created on/after this date")
        parser.add_argument(
            "--ensemble-covariate", action="store_true", help="Give Timestral the ensemble forecast as a covariate"
        )
        parser.add_argument("--csv", default="logs/timestral_backtest.csv", help="Per-slot results")

    def handle(self, *args, **options):
        prices = pd.Series(dict(PriceHistory.objects.values_list("date_time", "day_ahead")), dtype=float)
        prices.index = pd.DatetimeIndex(prices.index).tz_convert("UTC")
        prices = prices.sort_index()

        ff = pd.DataFrame(list(Forecasts.objects.values("id", "name", "created_at")))
        ff["created_at"] = pd.to_datetime(ff["created_at"], utc=True)
        ff["date"] = ff["created_at"].dt.tz_convert("GB").dt.normalize()
        runs = select_daily_training_forecasts(ff.set_index("id"))
        if options["start"]:
            runs = runs[runs["created_at"] >= pd.Timestamp(options["start"], tz="GB")]
        # Only runs with at least a day of outturn to score against.
        runs = runs[runs["created_at"] < prices.index[-1] - pd.Timedelta(days=1)].sort_values("created_at")

        forecaster = load_forecaster(options["model"])
        frames = []
        for run_id, run in runs.iterrows():
            run_obj = Forecasts.objects.get(id=run_id)
            cutoff, ensemble = first_unpriced_slot(run_obj, prices)
            try:
                ts = forecast_day_ahead(
                    options["model"], options["context_days"], run=run_obj, prices_before=cutoff,
                    forecaster=forecaster, with_ensemble=options["ensemble_covariate"],
                )
            except Exception as exc:
                self.stderr.write(f"{run['name']}: {exc}")
                continue
            ts.index = ts.index.tz_convert("UTC")
            df = pd.DataFrame(
                {
                    "timestral": ts["day_ahead"],
                    "ensemble": ensemble.reindex(ts.index),
                    "actual": prices.reindex(ts.index),
                }
            ).dropna()
            df["run"] = run["name"]
            df["lead_days"] = (df.index - run["created_at"]).total_seconds() / 86400
            frames.append(df)
            self.stdout.write(
                f"{run['name']}: {len(df)} scored slots  MAE ts={(df.timestral - df.actual).abs().mean():.1f} "
                f"ens={(df.ensemble - df.actual).abs().mean():.1f} £/MWh"
            )

        if not frames:
            self.stdout.write("No runs scored")
            return
        res = pd.concat(frames)
        res.rename_axis("date_time").to_csv(options["csv"])

        # Agile G p/kWh, the unit the site reports accuracy in.
        for col in ("timestral", "ensemble", "actual"):
            res[f"{col}_agile"] = day_ahead_to_agile(res[col].set_axis(res.index.tz_convert("GB")), region="G").to_numpy()
        res["lead"] = np.floor(res["lead_days"]).clip(lower=0).astype(int)

        def summarise(g):
            return pd.Series(
                {
                    "n": len(g),
                    "MAE_ts": (g.timestral_agile - g.actual_agile).abs().mean(),
                    "MAE_ens": (g.ensemble_agile - g.actual_agile).abs().mean(),
                    "RMSE_ts": np.sqrt(((g.timestral_agile - g.actual_agile) ** 2).mean()),
                    "RMSE_ens": np.sqrt(((g.ensemble_agile - g.actual_agile) ** 2).mean()),
                    "sd_ts": g.timestral_agile.std(),
                    "sd_ens": g.ensemble_agile.std(),
                    "sd_actual": g.actual_agile.std(),
                }
            )

        self.stdout.write(f"\nRuns scored: {res['run'].nunique()}  slots: {len(res)}  (Agile G, p/kWh)")
        self.stdout.write("By lead day:\n" + res.groupby("lead").apply(summarise).round(2).to_string())
        for label, mask in (("all leads", res["lead_days"] >= 0), ("lead >= 2 days", res["lead_days"] >= 2)):
            self.stdout.write(f"\n{label}:\n" + summarise(res[mask]).round(2).to_string())
        # Tail behaviour: where the outturn was in its bottom/top decile.
        lo, hi = res["actual_agile"].quantile([0.1, 0.9])
        for label, mask in (("actual bottom decile", res["actual_agile"] <= lo), ("actual top decile", res["actual_agile"] >= hi)):
            g = res[mask & (res["lead_days"] >= 2)]
            self.stdout.write(
                f"{label} (lead>=2d, n={len(g)}): mean actual {g.actual_agile.mean():.1f}  "
                f"ts {g.timestral_agile.mean():.1f}  ens {g.ensemble_agile.mean():.1f}  "
                f"MAE ts {(g.timestral_agile - g.actual_agile).abs().mean():.1f}  "
                f"ens {(g.ensemble_agile - g.actual_agile).abs().mean():.1f}"
            )
