import pandas as pd
from django.core.management.base import BaseCommand

from config.utils import day_ahead_to_agile
from prices.models import PriceHistory


class Command(BaseCommand):
    help = (
        "Recompute PriceHistory.day_ahead from the stored Agile (region G) price with the "
        "current conversion. Dry run unless --apply. Idempotent."
    )

    def add_arguments(self, parser):
        parser.add_argument("--apply", action="store_true", help="Write the recomputed values")
        parser.add_argument("--start", help="Only rows at or after this timestamp")
        parser.add_argument("--tolerance", type=float, default=0.001, help="Ignore changes smaller than this (GBP/MWh)")

    def handle(self, *args, **options):
        qs = PriceHistory.objects.exclude(agile__isnull=True).order_by("date_time")
        if options["start"]:
            qs = qs.filter(date_time__gte=pd.Timestamp(options["start"]))
        rows = list(qs)
        if not rows:
            self.stdout.write("No price rows")
            return

        agile = pd.Series([r.agile for r in rows], index=pd.DatetimeIndex([r.date_time for r in rows]))
        recomputed = day_ahead_to_agile(agile, reverse=True, region="G").to_numpy()

        changed = []
        for row, new in zip(rows, recomputed):
            if row.day_ahead is None or abs(row.day_ahead - new) > options["tolerance"]:
                changed.append((row, float(new)))

        self.stdout.write(f"Rows checked: {len(rows)} ({rows[0].date_time} to {rows[-1].date_time})")
        self.stdout.write(f"Rows to change: {len(changed)}")
        if changed:
            diffs = pd.Series(
                [new - (row.day_ahead or 0.0) for row, new in changed],
                index=pd.DatetimeIndex([row.date_time for row, _ in changed]),
            )
            by_month = diffs.groupby(diffs.index.tz_convert("GB").strftime("%Y-%m")).agg(["count", "mean", "min", "max"])
            self.stdout.write(f"Change in day_ahead by month (GBP/MWh):\n{by_month.round(2).tail(12)}")

        if not options["apply"]:
            self.stdout.write("Dry run: nothing written (pass --apply to write)")
            return

        for row, new in changed:
            row.day_ahead = new
        PriceHistory.objects.bulk_update([row for row, _ in changed], ["day_ahead"], batch_size=1000)
        self.stdout.write(f"Updated {len(changed)} rows")
