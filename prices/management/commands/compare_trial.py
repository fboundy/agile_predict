"""
Compare forecast quality before and after a code change, on published forecasts.

Built for the dynamic-range trial (see docs/MODEL_DYNAMIC_RANGE.md): the widened
training window ran on the dev server from 2026-08-16 while production stayed on
the previous build. This scores forecasts the model actually published against
prices that have since settled, which is the only genuinely out-of-sample
measurement available — the `build_holdout_data` holdout is not one.

Typical use on review day:

    # dev box: old code before the split, new code after
    python manage.py compare_trial --split 2026-08-16

    # prod, same window, old code throughout — the regime control
    fly ssh console --app prices --machine <web> \
        -C "python manage.py compare_trial --since 2026-08-16 --label prod"

Reading it: the before/after split on one box confounds the code change with the
weather and price regime of two different weeks. The prod run over the *same*
window as the dev "after" cell is what separates them. Treat a dev-only
before/after as suggestive, never conclusive.
"""

import json

import pandas as pd
from django.core.management.base import BaseCommand, CommandError

from prices.models import ForecastData, PriceHistory
from prices.model_metrics import format_report, stored_forecast_report


def _load(since=None, until=None, column="day_ahead"):
    qs = ForecastData.objects.filter(**{f"{column}__isnull": False})
    if since is not None:
        qs = qs.filter(forecast__created_at__gte=since)
    if until is not None:
        qs = qs.filter(forecast__created_at__lt=until)
    stored = pd.DataFrame(list(qs.values("date_time", column, "forecast__created_at")))
    if stored.empty:
        return stored
    stored = stored.rename(columns={"forecast__created_at": "created_at", column: "day_ahead"})
    stored["date_time"] = pd.to_datetime(stored["date_time"], utc=True)
    stored["created_at"] = pd.to_datetime(stored["created_at"], utc=True)
    return stored


def _settled_prices():
    ph = pd.DataFrame(list(PriceHistory.objects.values("date_time", "day_ahead")))
    if ph.empty:
        raise CommandError("No PriceHistory rows; nothing to score against.")
    ph["date_time"] = pd.to_datetime(ph["date_time"], utc=True)
    return ph.set_index("date_time")


class Command(BaseCommand):
    help = "Score published forecasts against settled prices, optionally split before/after a date."

    def add_arguments(self, parser):
        parser.add_argument("--since", help="Only forecasts created on/after this date (YYYY-MM-DD).")
        parser.add_argument("--until", help="Only forecasts created before this date (YYYY-MM-DD).")
        parser.add_argument(
            "--split",
            help="Report twice, before and after this date (YYYY-MM-DD), plus a delta table.",
        )
        parser.add_argument(
            "--min-horizon", type=float, default=2.0,
            help="Minimum forecast horizon in days (default 2; below that the pipeline blends GB60 actuals).",
        )
        parser.add_argument("--label", default="", help="Label for the output, e.g. 'prod'.")
        parser.add_argument(
            "--column", default="day_ahead",
            choices=["day_ahead", "day_ahead_corrected"],
            help="Which stored series to score. day_ahead_corrected is the "
                 "post-processed forecast (see prices/postprocess.py).",
        )
        parser.add_argument("--json", action="store_true", help="Emit JSON instead of a formatted report.")

    def handle(self, *args, **options):
        prices = _settled_prices()
        min_horizon = options["min_horizon"]
        label = options["label"]
        column = options["column"]

        def report(since, until, name):
            stored = _load(since, until, column)
            if stored.empty:
                self.stdout.write(f"{name}: no published forecasts in window")
                return None
            rep = stored_forecast_report(stored, prices, min_horizon_days=min_horizon)
            if rep is None:
                self.stdout.write(f"{name}: too few settled rows to report ({len(stored)} published)")
                return None
            rep["window"] = name
            rep["n_published"] = len(stored)
            return rep

        def as_ts(value):
            return pd.Timestamp(value, tz="UTC") if value else None

        results = []
        if options["split"]:
            split = as_ts(options["split"])
            results.append(report(as_ts(options["since"]), split, f"before {options['split']}"))
            results.append(report(split, as_ts(options["until"]), f"from {options['split']}"))
        else:
            results.append(report(as_ts(options["since"]), as_ts(options["until"]), "window"))

        results = [r for r in results if r]
        if not results:
            raise CommandError("No reportable windows.")

        if options["json"]:
            self.stdout.write(json.dumps({"label": label, "reports": results}, indent=2, default=str))
            return

        for rep in results:
            suffix = f" {label}" if label else ""
            self.stdout.write("")
            self.stdout.write(format_report(rep, f"[{rep['window']}{suffix}, >={min_horizon}d, {column}]"))

        if len(results) == 2:
            before, after = results
            self.stdout.write("")
            self.stdout.write("Delta (after - before):")
            rows = [
                ("sd_ratio", "sd_ratio", ">9.3f"), ("slope", "slope", ">9.3f"),
                ("rmse", "rmse", ">9.2f"), ("mae", "mae", ">9.2f"),
                ("low_bias", "low_bias", ">+9.2f"), ("high_bias", "high_bias", ">+9.2f"),
            ]
            for name, key, spec in rows:
                b, a = before.get(key), after.get(key)
                if b is None or a is None or pd.isna(b) or pd.isna(a):
                    continue
                self.stdout.write(f"  {name:<12s} {b:{spec}} -> {a:{spec}}   ({a - b:+.3f})")
            def num(v, spec=".3f"):
                return "n/a" if v is None or pd.isna(v) else format(v, spec)

            for band in before.get("bands", {}):
                b = before["bands"][band]
                a = after["bands"][band]
                self.stdout.write(
                    f"  {band:<12s} recall {num(b['recall'])} -> {num(a['recall'])}   "
                    f"precision {num(b['precision'])} -> {num(a['precision'])}   "
                    f"(n_actual {b['n_actual']} -> {a['n_actual']})"
                )
                if not a["n_actual"]:
                    self.stdout.write(
                        f"  {'':<12s} WARNING: no {band} events settled in the 'after' window — "
                        "this band is untestable here, not unchanged."
                    )
            self.stdout.write("")
            self.stdout.write(
                "NOTE: a before/after split on one box confounds the code change with the\n"
                "weather and price regime of two different weeks. Run the same command on\n"
                "prod over the 'after' window to separate them."
            )
