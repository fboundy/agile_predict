import pandas as pd
from django.core.management.base import BaseCommand

from prices.forecast_eval import CALIBRATORS, add_backtest_column, calibrate, load_ensemble_frame, score


class Command(BaseCommand):
    help = (
        "Score stored forecasts on user-facing metrics (cheapest/priciest windows, negative prices, "
        "tails, within-day ranking) as well as MAE/RMSE, optionally with out-of-sample spread calibration."
    )

    def add_arguments(self, parser):
        parser.add_argument("--since", help="Load runs created on/after this date (calibration history included)")
        parser.add_argument("--score-since", help="Only score runs created on/after this date")
        parser.add_argument("--min-lead", type=float, default=2.0, help="Only score slots at least this many days ahead")
        parser.add_argument(
            "--backtest", action="append", default=[], metavar="NAME=CSV",
            help="Add a timestral_backtest CSV as a forecast column (repeatable)",
        )
        parser.add_argument("--daily-only", action="store_true", help="Score only runs present in every backtest CSV")
        parser.add_argument("--calibrate", default="", help=f"Comma-separated methods: {', '.join(CALIBRATORS)}")
        parser.add_argument("--window-days", type=int, default=28, help="Calibration trailing window")
        parser.add_argument("--csv", help="Write the per-slot frame (with calibrated columns) here")

    def handle(self, *args, **options):
        df = load_ensemble_frame(options["since"])
        cols = ["ensemble"]
        for spec in options["backtest"]:
            name, path = spec.split("=", 1)
            df = add_backtest_column(df, path, name)
            cols.append(name)
        if options["daily_only"] and len(cols) > 1:
            df = df.dropna(subset=cols[1:])

        for method in filter(None, options["calibrate"].split(",")):
            for col in list(cols):
                if "+" in col:
                    continue
                name = f"{col}+{method}"
                df[name] = calibrate(df, col, method, options["window_days"])
                cols.append(name)

        scored = df[df["lead_days"] >= options["min_lead"]]
        if options["score_since"]:
            scored = scored[scored["created_at"] >= pd.Timestamp(options["score_since"], tz="GB")]
        table = score(scored, cols)
        self.stdout.write(
            f"Agile G, p/kWh. lead >= {options['min_lead']} d; {table.attrs['runs']} runs, "
            f"{table.attrs['slots']} slots, {table.attrs['negative_slots']} negative-price slots"
        )
        self.stdout.write(table.T.round(3).to_string())

        if options["csv"]:
            df.to_csv(options["csv"], index=False)
