"""
Backfill fr_wind and fr_rad on historical ForecastData using Open-Meteo archive.
Uses the archive API (archive-api.open-meteo.com) for past dates and the
forecast API for any future dates still in the data.
"""
import requests
import pandas as pd

from django.core.management.base import BaseCommand

from prices.models import ForecastData

ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
FR_LAT, FR_LON = 47.0, 2.0
BATCH_SIZE = 2000


def _fetch_archive(start_date, end_date):
    """Fetch hourly fr_wind + fr_rad from Open-Meteo archive for France."""
    resp = requests.get(ARCHIVE_URL, params={
        "latitude":  FR_LAT,
        "longitude": FR_LON,
        "start_date": start_date,
        "end_date":   end_date,
        "hourly": "wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms",
        "timeformat": "unixtime",
        "timezone": "UTC",
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    times = pd.to_datetime(data["hourly"]["time"], unit="s", utc=True)
    return pd.DataFrame({
        "fr_wind": data["hourly"]["wind_speed_10m"],
        "fr_rad":  data["hourly"]["shortwave_radiation"],
    }, index=times)


def _fetch_forecast_fr():
    """Fetch 16-day Open-Meteo forecast for France (for any near-future rows)."""
    resp = requests.get(FORECAST_URL, params={
        "latitude":  FR_LAT,
        "longitude": FR_LON,
        "hourly": "wind_speed_10m,shortwave_radiation",
        "wind_speed_unit": "ms",
        "timeformat": "unixtime",
        "timezone": "UTC",
        "forecast_days": 16,
    }, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    times = pd.to_datetime(data["hourly"]["time"], unit="s", utc=True)
    return pd.DataFrame({
        "fr_wind": data["hourly"]["wind_speed_10m"],
        "fr_rad":  data["hourly"]["shortwave_radiation"],
    }, index=times)


class Command(BaseCommand):
    help = "Backfill fr_wind and fr_rad on ForecastData from Open-Meteo archive"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Overwrite existing values")

    def handle(self, *args, **options):
        force = options["force"]
        qs = ForecastData.objects.all() if force else ForecastData.objects.filter(fr_wind__isnull=True)
        total = qs.count()
        self.stdout.write(f"Rows to backfill: {total}")
        if total == 0:
            self.stdout.write("Nothing to do.")
            return

        bounds = qs.aggregate(
            min_dt=__import__("django.db.models", fromlist=["Min"]).Min("date_time"),
            max_dt=__import__("django.db.models", fromlist=["Max"]).Max("date_time"),
        )
        min_dt = pd.Timestamp(bounds["min_dt"]).tz_convert("UTC")
        max_dt = pd.Timestamp(bounds["max_dt"]).tz_convert("UTC")
        today = pd.Timestamp.now(tz="UTC").normalize()

        self.stdout.write(f"Date range: {min_dt.date()} → {max_dt.date()}")

        # Build combined hourly index from archive + forecast
        frames = []
        if min_dt.date() < today.date():
            archive_end = min(max_dt, today - pd.Timedelta("1d"))
            self.stdout.write(f"Fetching archive: {min_dt.date()} → {archive_end.date()}")
            frames.append(_fetch_archive(str(min_dt.date()), str(archive_end.date())))

        if max_dt >= today:
            self.stdout.write("Fetching 16-day forecast for future rows…")
            frames.append(_fetch_forecast_fr())

        if not frames:
            self.stdout.write("No data fetched — nothing to do.")
            return

        weather = pd.concat(frames).sort_index()
        weather = weather[~weather.index.duplicated(keep="last")]

        # Process in batches
        updated = 0
        pks = list(qs.values_list("pk", "date_time"))
        self.stdout.write(f"Matching {len(pks)} rows to weather index…")

        batch = []
        for pk, dt in pks:
            ts = pd.Timestamp(dt).tz_convert("UTC")
            # Nearest-hour lookup (weather is hourly)
            ts_hour = ts.floor("h")
            if ts_hour not in weather.index:
                # try rounding to nearest hour
                ts_hour = ts.round("h")
            row = weather.get(ts_hour) if ts_hour in weather.index else None
            if row is None:
                # find nearest
                idx = weather.index.get_indexer([ts], method="nearest")[0]
                if idx < 0:
                    continue
                row = weather.iloc[idx]

            obj = ForecastData(pk=pk)
            obj.fr_wind = float(row["fr_wind"]) if pd.notna(row["fr_wind"]) else None
            obj.fr_rad  = float(row["fr_rad"])  if pd.notna(row["fr_rad"])  else None
            batch.append(obj)

            if len(batch) >= BATCH_SIZE:
                ForecastData.objects.bulk_update(batch, ["fr_wind", "fr_rad"])
                updated += len(batch)
                batch = []
                self.stdout.write(f"  {updated}/{total}")

        if batch:
            ForecastData.objects.bulk_update(batch, ["fr_wind", "fr_rad"])
            updated += len(batch)

        self.stdout.write(self.style.SUCCESS(f"Done — updated {updated} rows."))
