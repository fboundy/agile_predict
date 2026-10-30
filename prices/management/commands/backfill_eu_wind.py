"""
Backfill eu_wind_cf on historical ForecastData from Open-Meteo's Previous Runs API, so training
rows see the continental wind *as it was forecast* (not the outturn, as backfill_fr_weather does).
Rows still in the future take the live forecast.
"""
import pandas as pd

from django.core.management.base import BaseCommand

from config.utils import get_open_meteo_eu_wind
from prices.models import ForecastData

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
BATCH_SIZE = 2000


class Command(BaseCommand):
    help = "Backfill eu_wind_cf on ForecastData from Open-Meteo previous ECMWF runs"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Overwrite existing values")

    def handle(self, *args, **options):
        qs = ForecastData.objects.all() if options["force"] else ForecastData.objects.filter(eu_wind_cf__isnull=True)
        rows = pd.DataFrame(qs.values_list("pk", "date_time"), columns=["pk", "date_time"])
        if rows.empty:
            self.stdout.write("Nothing to backfill.")
            return

        # shortcut: one 24-48 h lead (previous_day1) for every row — it matches the 22-46 h training
        # window; holdout/long-horizon rows get a too-good value. Use previous_dayN by row horizon if
        # the training window is widened.
        cf = get_open_meteo_eu_wind(
            url=PREVIOUS_RUNS_URL,
            variable="wind_speed_100m_previous_day1",
            start_date=rows.date_time.min().strftime("%Y-%m-%d"),
            end_date=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
        )
        cf = pd.concat([cf, get_open_meteo_eu_wind()]).sort_index()
        cf = cf[~cf.index.duplicated(keep="first")]
        if cf.empty:
            self.stdout.write(self.style.ERROR("No data fetched — nothing to do."))
            return

        rows["eu_wind_cf"] = cf.reindex(pd.DatetimeIndex(rows.date_time), method="nearest").to_numpy()
        updated = 0
        for start in range(0, len(rows), BATCH_SIZE):
            batch = [ForecastData(pk=r.pk, eu_wind_cf=float(r.eu_wind_cf)) for r in rows.iloc[start:start + BATCH_SIZE].itertuples()]
            ForecastData.objects.bulk_update(batch, ["eu_wind_cf"])
            updated += len(batch)
            self.stdout.write(f"  {updated}/{len(rows)}")
        self.stdout.write(self.style.SUCCESS(f"Done — updated {updated} rows."))
