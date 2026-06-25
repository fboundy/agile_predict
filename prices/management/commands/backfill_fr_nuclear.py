"""
Backfill fr_nuclear on historical ForecastData using RTE eco2mix historical dataset.
For each forecast, looks up the French nuclear value at forecast creation time
(matching production behaviour: current value forward-filled to all future slots).
"""
import requests
import pandas as pd

from django.core.management.base import BaseCommand

from prices.models import ForecastData, Forecasts

RTE_URL = "https://opendata.reseaux-energies.fr/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/records"
PAGE_SIZE = 100
BATCH_SIZE = 2000


def fetch_rte_nuclear_history(start_dt, end_dt):
    """
    Paginate the eco2mix consolidated dataset for the target date range.
    ODS API v2.1 where clause doesn't support timestamp filters (returns 400),
    so we use total_count to calculate the offset into the descending-sorted
    dataset that corresponds to our start date, then filter in Python.
    Returns a 30-min resampled UTC-indexed Series named 'fr_nuclear'.
    """
    start_ts = pd.Timestamp(start_dt).tz_convert("UTC")
    end_ts   = pd.Timestamp(end_dt).tz_convert("UTC")

    # ODS API has a maximum offset limit, so we fetch newest-first (desc)
    # and stop once records go older than our start date.
    records = []
    offset = 0
    while True:
        resp = requests.get(RTE_URL, params={
            "select":   "date_heure,nucleaire",
            "order_by": "date_heure desc",
            "limit":    PAGE_SIZE,
            "offset":   offset,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break
        records.extend(batch)
        offset += PAGE_SIZE
        # Stop once the oldest record in this batch is before our start
        oldest_ts = pd.Timestamp(batch[-1]["date_heure"]).tz_convert("UTC")
        if oldest_ts < start_ts:
            break

    if not records:
        return pd.Series(dtype=float, name="fr_nuclear")

    df = pd.DataFrame(records)
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df = df.set_index("date_heure").sort_index()
    # Filter to actual date range
    df = df[(df.index >= start_ts) & (df.index <= end_ts)]
    df["nucleaire"] = pd.to_numeric(df["nucleaire"], errors="coerce")
    df = df.dropna(subset=["nucleaire"])
    if df.empty:
        return pd.Series(dtype=float, name="fr_nuclear")

    s = df["nucleaire"].resample("30min").mean().ffill()
    s.name = "fr_nuclear"
    return s


class Command(BaseCommand):
    help = "Backfill fr_nuclear on ForecastData from RTE eco2mix historical data"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Overwrite existing values")

    def handle(self, *args, **options):
        force = options["force"]
        qs = ForecastData.objects.all() if force else ForecastData.objects.filter(fr_nuclear__isnull=True)
        total = qs.count()
        self.stdout.write(f"Rows to backfill: {total}")
        if total == 0:
            self.stdout.write("Nothing to do.")
            return

        # Fetch all forecasts that have null fr_nuclear rows
        forecast_ids = set(qs.values_list("forecast_id", flat=True).distinct())
        forecasts = list(Forecasts.objects.filter(pk__in=forecast_ids).order_by("created_at"))
        self.stdout.write(f"Forecasts to process: {len(forecasts)}")

        if not forecasts:
            return

        min_dt = min(f.created_at for f in forecasts) - pd.Timedelta("1h")
        max_dt = max(f.created_at for f in forecasts) + pd.Timedelta("1h")
        self.stdout.write(f"Fetching RTE nuclear: {min_dt.date()} → {max_dt.date()} …")
        nuclear_ts = fetch_rte_nuclear_history(min_dt, max_dt)
        self.stdout.write(f"  Got {len(nuclear_ts)} data points")

        if nuclear_ts.empty:
            self.stdout.write(self.style.ERROR("No RTE data — aborting."))
            return

        updated = 0
        batch = []
        n_forecasts = len(forecasts)

        for idx, forecast in enumerate(forecasts, 1):
            created = pd.Timestamp(forecast.created_at).tz_convert("UTC")
            # Get nuclear value at forecast creation time (nearest available)
            loc_idx = nuclear_ts.index.get_indexer([created], method="nearest")[0]
            nuclear_val = float(nuclear_ts.iloc[loc_idx]) if loc_idx >= 0 else None

            rows = list(
                ForecastData.objects.filter(forecast=forecast, fr_nuclear__isnull=True)
                .only("pk", "fr_nuclear")
            )
            for row in rows:
                row.fr_nuclear = nuclear_val
                batch.append(row)

            if len(batch) >= BATCH_SIZE:
                ForecastData.objects.bulk_update(batch, ["fr_nuclear"])
                updated += len(batch)
                batch = []

            if idx % 50 == 0 or idx == n_forecasts:
                self.stdout.write(f"  {idx}/{n_forecasts} forecasts, {updated} rows updated")

        if batch:
            ForecastData.objects.bulk_update(batch, ["fr_nuclear"])
            updated += len(batch)

        self.stdout.write(self.style.SUCCESS(f"Done — updated {updated} rows."))
