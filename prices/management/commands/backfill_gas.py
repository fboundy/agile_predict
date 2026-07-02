"""
Backfill gas_availability on historical ForecastData using the BMRS availability
evolution endpoint (same source as nuclear), summing CCGT + OCGT for each forecast date.

Efficiency: fetches the full publish-time history for each unique target date once,
then resolves each forecast's value locally without extra API calls.
"""
import requests
import pandas as pd
from collections import defaultdict

from django.core.management.base import BaseCommand

from prices.models import ForecastData, Forecasts

BATCH_SIZE = 2000
EVOLUTION_URL = "https://data.elexon.co.uk/bmrs/api/v1/forecast/availability/daily/evolution"


def _fetch_evolution(forecast_date_str):
    """Return dict of {publishTime (UTC Timestamp): total_mw} for a given target date."""
    totals = defaultdict(float)
    for fuel in ("CCGT", "OCGT"):
        try:
            resp = requests.get(
                EVOLUTION_URL,
                params={"fuelType": fuel, "forecastDate": forecast_date_str, "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            for r in resp.json().get("data", []):
                pt = pd.Timestamp(r["publishTime"], tz="UTC")
                totals[pt] += float(r["outputUsable"])
        except Exception:
            pass
    return totals


class Command(BaseCommand):
    help = "Backfill gas_availability (CCGT+OCGT MW) on ForecastData from BMRS evolution endpoint"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Overwrite existing values")

    def handle(self, *args, **options):
        force = options["force"]
        qs = ForecastData.objects.all() if force else ForecastData.objects.filter(gas_availability__isnull=True)
        total_rows = qs.count()
        self.stdout.write(f"Rows to backfill: {total_rows}")
        if total_rows == 0:
            self.stdout.write("Nothing to do.")
            return

        forecast_ids = set(qs.values_list("forecast_id", flat=True).distinct())
        forecasts = list(Forecasts.objects.filter(pk__in=forecast_ids).order_by("created_at"))

        # Collect all unique forecast target dates (UTC date strings)
        all_dates = set(
            pd.Timestamp(dt).tz_convert("UTC").normalize().date().isoformat()
            for dt in qs.values_list("date_time", flat=True).distinct()
        )
        self.stdout.write(f"Fetching evolution for {len(all_dates)} unique target dates...")

        # Pre-fetch full evolution history per target date
        evolution = {}  # date_str -> sorted list of (publishTime, total_mw)
        for i, date_str in enumerate(sorted(all_dates), 1):
            totals = _fetch_evolution(date_str)
            if totals:
                evolution[date_str] = sorted(totals.items())
            if i % 20 == 0:
                self.stdout.write(f"  {i}/{len(all_dates)} dates fetched")
        self.stdout.write(f"  Done — {len(evolution)} dates have data")

        def _gas_at(date_str, created_at_utc):
            pts = evolution.get(date_str)
            if not pts:
                return None
            known = [(pt, mw) for pt, mw in pts if pt <= created_at_utc]
            src = known if known else pts
            return src[-1][1]

        updated = 0
        batch = []
        n = len(forecasts)

        for idx, forecast in enumerate(forecasts, 1):
            created_at = pd.Timestamp(forecast.created_at).tz_convert("UTC")
            slot_filter = {"forecast": forecast} if force else {"forecast": forecast, "gas_availability__isnull": True}
            rows = list(ForecastData.objects.filter(**slot_filter).only("pk", "date_time", "gas_availability"))

            for row in rows:
                date_str = pd.Timestamp(row.date_time).tz_convert("UTC").normalize().date().isoformat()
                val = _gas_at(date_str, created_at)
                if val is not None:
                    row.gas_availability = val
                batch.append(row)

            if len(batch) >= BATCH_SIZE:
                ForecastData.objects.bulk_update(batch, ["gas_availability"])
                updated += len(batch)
                batch = []

            if idx % 50 == 0 or idx == n:
                self.stdout.write(f"  {idx}/{n} forecasts, {updated} rows updated")

        if batch:
            ForecastData.objects.bulk_update(batch, ["gas_availability"])
            updated += len(batch)

        self.stdout.write(self.style.SUCCESS(f"Done — updated {updated} rows."))
