"""
Backfill opmr_surplus on historical ForecastData using the NESO OPMR dataset.

For each forecast slot, computes the per-slot surplus:
  gen_availability + max_ic_import - slot_demand - opmr_total - constrained_plant

This replaces the old formula (National Surplus, anchored to peak demand) with a
value that reflects the actual margin at each half-hour period, removing the
systematic bias that peak-anchoring introduced on high-volatility days.
"""
import requests
import pandas as pd

from django.core.management.base import BaseCommand

from prices.models import ForecastData, Forecasts

OPMR_URL = "https://api.neso.energy/api/3/action/datastore_search"
RESOURCE_ID = "0eede912-8820-4c66-a58a-f7436d36b95f"
PAGE_SIZE = 500
BATCH_SIZE = 2000


def fetch_opmr_history(earliest_date, latest_publish):
    """
    Fetch all OPMR records with target Date >= earliest_date.
    Returns a DataFrame with columns:
      target_date, publish_date, gen_availability, max_ic_import, opmr_total, constrained_plant
    """
    r0 = requests.get(OPMR_URL, params={
        "resource_id": RESOURCE_ID,
        "limit": 1,
        "sort": "_id asc",
    }, timeout=15)
    r0.raise_for_status()
    total = r0.json().get("result", {}).get("total", 0)

    today_est = pd.Timestamp(earliest_date).normalize()
    dataset_start = pd.Timestamp("2021-03-14", tz="UTC")
    days_before = max((today_est - dataset_start).days - 90, 0)
    start_offset = max(0, int(days_before * 14) - 1500)

    all_records = []
    offset = start_offset
    while offset < total + PAGE_SIZE:
        resp = requests.get(OPMR_URL, params={
            "resource_id": RESOURCE_ID,
            "limit": PAGE_SIZE,
            "sort": "_id asc",
            "offset": offset,
        }, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("result", {}).get("records", [])
        if not batch:
            break
        all_records.extend(batch)
        last_pub = pd.Timestamp(batch[-1]["Publish Date"], tz="UTC")
        if last_pub > pd.Timestamp(latest_publish).tz_convert("UTC") + pd.Timedelta("7d"):
            break
        offset += PAGE_SIZE

    if not all_records:
        return pd.DataFrame(columns=[
            "target_date", "publish_date",
            "gen_availability", "max_ic_import", "opmr_total", "constrained_plant", "national_surplus",
        ])

    df = pd.DataFrame(all_records)
    df["target_date"]         = pd.to_datetime(df["Date"], utc=True).dt.normalize()
    df["publish_date"]        = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()
    df["gen_availability"]    = pd.to_numeric(df["Generator Availability"], errors="coerce")
    df["max_ic_import"]       = pd.to_numeric(df["Maximum I/C Import"], errors="coerce")
    df["opmr_total"]          = pd.to_numeric(df["OPMR total"], errors="coerce")
    df["constrained_plant"]   = pd.to_numeric(df["Constrained Plant"], errors="coerce").fillna(0)
    df["national_surplus"]    = pd.to_numeric(df["National Surplus"], errors="coerce")

    return df[["target_date", "publish_date",
               "gen_availability", "max_ic_import",
               "opmr_total", "constrained_plant", "national_surplus"]].dropna(
        subset=["gen_availability", "max_ic_import", "opmr_total"]
    )


class Command(BaseCommand):
    help = "Backfill opmr_surplus on ForecastData using per-slot gen_availability - slot_demand - opmr_total"

    def add_arguments(self, parser):
        parser.add_argument("--force", action="store_true", help="Overwrite existing values")

    def handle(self, *args, **options):
        force = options["force"]
        qs = ForecastData.objects.all() if force else ForecastData.objects.filter(opmr_surplus__isnull=True)
        total_rows = qs.count()
        self.stdout.write(f"Rows to backfill: {total_rows}")
        if total_rows == 0:
            self.stdout.write("Nothing to do.")
            return

        forecast_ids = set(qs.values_list("forecast_id", flat=True).distinct())
        forecasts = list(Forecasts.objects.filter(pk__in=forecast_ids).order_by("created_at"))
        self.stdout.write(f"Forecasts to process: {len(forecasts)}")
        if not forecasts:
            return

        earliest_target = qs.order_by("date_time").values_list("date_time", flat=True).first()
        latest_created  = max(f.created_at for f in forecasts)
        self.stdout.write(f"Fetching OPMR: from {pd.Timestamp(earliest_target).date()} ...")
        opmr_df = fetch_opmr_history(earliest_target, latest_created)
        self.stdout.write(
            f"  Got {len(opmr_df)} OPMR records "
            f"({opmr_df['target_date'].min().date() if len(opmr_df) else '?'} "
            f"to {opmr_df['target_date'].max().date() if len(opmr_df) else '?'})"
        )

        if opmr_df.empty:
            self.stdout.write(self.style.ERROR("No OPMR data fetched — aborting."))
            return

        updated = 0
        batch = []
        n_forecasts = len(forecasts)

        for idx, forecast in enumerate(forecasts, 1):
            created = pd.Timestamp(forecast.created_at).tz_convert("UTC")

            # OPMR values known at forecast creation time
            known = opmr_df[opmr_df["publish_date"] <= created.normalize()]
            if known.empty:
                known = opmr_df

            # Most recently published components per target date
            latest_per_day = (
                known.sort_values("publish_date")
                .drop_duplicates(subset="target_date", keep="last")
                .set_index("target_date")
            )

            slot_filter = {"forecast": forecast} if force else {"forecast": forecast, "opmr_surplus__isnull": True}
            slot_qs = ForecastData.objects.filter(**slot_filter).only(
                "pk", "date_time", "opmr_surplus", "opmr_national_surplus", "demand"
            )
            rows = list(slot_qs)

            for row in rows:
                slot_date = pd.Timestamp(row.date_time).tz_convert("UTC").normalize()
                available = latest_per_day[latest_per_day.index <= slot_date]
                if available.empty:
                    fwd = latest_per_day[latest_per_day.index >= slot_date]
                    opmr_row = fwd.iloc[0] if not fwd.empty else None
                else:
                    opmr_row = available.iloc[-1]

                if opmr_row is not None and row.demand is not None:
                    row.opmr_surplus = float(
                        opmr_row["gen_availability"]
                        + opmr_row["max_ic_import"]
                        - row.demand
                        - opmr_row["opmr_total"]
                        - opmr_row["constrained_plant"]
                    )
                    row.opmr_national_surplus = float(opmr_row["national_surplus"])
                batch.append(row)

            if len(batch) >= BATCH_SIZE:
                ForecastData.objects.bulk_update(batch, ["opmr_surplus", "opmr_national_surplus"])
                updated += len(batch)
                batch = []

            if idx % 50 == 0 or idx == n_forecasts:
                self.stdout.write(f"  {idx}/{n_forecasts} forecasts, {updated} rows updated")

        if batch:
            ForecastData.objects.bulk_update(batch, ["opmr_surplus", "opmr_national_surplus"])
            updated += len(batch)

        self.stdout.write(self.style.SUCCESS(f"Done — updated {updated} rows."))
