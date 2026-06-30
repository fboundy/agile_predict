"""
Backfill fr_nuclear on historical ForecastData using ENTSO-E A75 actual
generation data (psrType=B14 = Nuclear) for France.
eco2mix consolidated dataset has a 5-month+ delay; ENTSO-E has full history.
For each forecast, looks up the French nuclear value at forecast creation time
(matching production behaviour: current value forward-filled to all future slots).
"""
import requests
import pandas as pd
from xml.etree import ElementTree as ET

from django.conf import settings
from django.core.management.base import BaseCommand

from prices.models import ForecastData, Forecasts

ENTSOE_URL = "https://web-api.tp.entsoe.eu/api"
BATCH_SIZE = 2000


def _parse_entsoe_qty(xml_text):
    """Parse ENTSO-E generation XML into UTC-indexed Series of quantities."""
    root = ET.fromstring(xml_text)
    if "Acknowledgement" in root.tag:
        return pd.Series(dtype=float)
    records = []
    for ts in root.findall(".//{*}TimeSeries"):
        for period in ts.findall("{*}Period"):
            start_el = period.find("{*}timeInterval/{*}start")
            res_el   = period.find("{*}resolution")
            if start_el is None or res_el is None:
                continue
            start_ts = pd.Timestamp(start_el.text, tz="UTC")
            minutes  = {"PT60M": 60, "PT30M": 30, "PT15M": 15}.get(res_el.text, 60)
            for point in period.findall("{*}Point"):
                pos_el = point.find("{*}position")
                qty_el = point.find("{*}quantity")
                if pos_el is None or qty_el is None:
                    continue
                ts_val = start_ts + pd.Timedelta(minutes=minutes * (int(pos_el.text) - 1))
                records.append({"ts": ts_val, "qty": float(qty_el.text)})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).set_index("ts")["qty"].sort_index()


def fetch_entsoe_fr_nuclear(start_dt, end_dt):
    """
    Fetch ENTSO-E A75 actual nuclear generation for France in monthly chunks.
    Returns a 30-min resampled UTC-indexed Series named 'fr_nuclear'.
    """
    token = getattr(settings, "ENTSOE_API_KEY", "")
    if not token:
        return pd.Series(dtype=float, name="fr_nuclear")

    start_ts = pd.Timestamp(start_dt).tz_convert("UTC").normalize()
    end_ts   = pd.Timestamp(end_dt).tz_convert("UTC").normalize() + pd.Timedelta("1d")

    all_series = []
    chunk_start = start_ts
    while chunk_start < end_ts:
        chunk_end = min(chunk_start + pd.Timedelta("32d"), end_ts)
        params = {
            "securityToken": token,
            "documentType":  "A75",
            "processType":   "A16",
            "in_Domain":     "10YFR-RTE------C",
            "psrType":       "B14",
            "periodStart":   chunk_start.strftime("%Y%m%d%H%M"),
            "periodEnd":     chunk_end.strftime("%Y%m%d%H%M"),
        }
        resp = requests.get(ENTSOE_URL, params=params, timeout=120)
        resp.raise_for_status()
        s = _parse_entsoe_qty(resp.text)
        if not s.empty:
            all_series.append(s)
        chunk_start = chunk_end

    if not all_series:
        return pd.Series(dtype=float, name="fr_nuclear")

    combined = pd.concat(all_series).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.resample("30min").mean().ffill()
    combined.name = "fr_nuclear"
    return combined


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
        self.stdout.write(f"Fetching ENTSO-E A75 FR nuclear: {min_dt.date()} → {max_dt.date()} …")
        nuclear_ts = fetch_entsoe_fr_nuclear(min_dt, max_dt)
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
