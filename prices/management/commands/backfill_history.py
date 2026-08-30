"""Backfill the `History` table from NESO/Open-Meteo archives.

DEV ONLY. The owner's decision (2026-08-27) is to hold this data on the dev box so
production's database stays small. `History` is not read by the forecast pipeline;
it exists so that plunge/spike behaviour can be studied across seasons rather than
the two months of `ForecastData` we currently have. See docs/MODEL_DYNAMIC_RANGE.md.

The guard is `.dockerignore`: this file is excluded from the build context, so it
is simply not present in the production image and cannot be run there by any route.
`prices/tests.py::DockerignoreGuardTests` asserts that entry survives.

Why a new command rather than `full_hist`:

* `full_hist` calls `get_latest_history()`, whose demand sources are the *per-year*
  "Historic Demand Data" resources — and only the 2023 and 2024 ids are listed.
  There is no 2025 or 2026 source, so any window past 2024 loses demand entirely,
  and the frame's final `dropna()` then empties the whole result.
* The NESO SQL endpoint caps a single response at a few thousand rows regardless of
  the `LIMIT` asked for, so a one-shot fetch silently truncates. This command
  paginates with LIMIT/OFFSET instead.

Column semantics are chosen to match what the forecast features mean, so historic
rows are comparable with `ForecastData`:

    demand      ND                       (transmission demand, per-year resource)
    solar       SOLAR                    (generation mix)
    bm_wind     WIND                     (transmission-connected wind)
    total_wind  WIND + WIND_EMB          (so emb_wind = total_wind - bm_wind, the
                                          derivation views.py already uses)
    nuclear     NUCLEAR                  (generation mix)
    temp_2m / wind_10m / rad             (Open-Meteo archive, resampled to 30min)
    gas_ttf                              (existing get_gas_ttf_history)
"""

import time
from urllib import parse

import pandas as pd
import requests
from django.core.management.base import BaseCommand

from config.utils import get_gas_ttf_history
from prices.models import History

SQL_URL = "https://api.neso.energy/api/3/action/datastore_search_sql"

# "Historic Demand Data" is published one resource per calendar year.
DEMAND_RESOURCES = {
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
    2026: "8a4a771c-3929-4e56-93ad-cdf13219dea5",
}
GEN_MIX_RESOURCE = "f93d1835-75bc-43e5-84ad-12472b180a98"

PAGE = 5000
REQUIRED = ["total_wind", "bm_wind", "solar", "temp_2m", "wind_10m", "rad", "demand"]


def _sql_page(resource_id, where, order_col, limit, offset):
    sql = (f'SELECT * FROM "{resource_id}" WHERE {where} '
           f'ORDER BY "{order_col}" ASC LIMIT {limit} OFFSET {offset}')
    r = requests.get(SQL_URL, params=parse.urlencode({"sql": sql}), timeout=180)
    r.raise_for_status()
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"NESO SQL failed: {str(body.get('error'))[:200]}")
    return body["result"]["records"]


def fetch_paginated(resource_id, where, order_col, label, stdout):
    """Page through a datastore resource; the endpoint truncates single responses."""
    rows, offset = [], 0
    while True:
        recs = _sql_page(resource_id, where, order_col, PAGE, offset)
        if not recs:
            break
        rows.extend(recs)
        stdout.write(f"    {label}: {len(rows)} rows")
        if len(recs) < PAGE:
            break
        offset += len(recs)
        time.sleep(0.3)
    return pd.DataFrame(rows)


class Command(BaseCommand):
    help = "Backfill History from NESO/Open-Meteo archives (dev only)."

    def add_arguments(self, parser):
        parser.add_argument("--start", default="2023-07-01")
        parser.add_argument("--end", default=None, help="default: now")
        parser.add_argument("--dry-run", action="store_true",
                            help="fetch and report coverage without writing")

    def handle(self, *args, **options):
        start = pd.Timestamp(options["start"], tz="UTC")
        end = (pd.Timestamp(options["end"], tz="UTC") if options["end"]
               else pd.Timestamp.now(tz="UTC"))
        self.stdout.write(f"Backfilling History {start.date()} -> {end.date()}")

        # ---- demand, one resource per year ----
        self.stdout.write("  fetching demand (per-year resources)...")
        frames = []
        for year, rid in sorted(DEMAND_RESOURCES.items()):
            if year < start.year or year > end.year:
                continue
            where = f""""SETTLEMENT_DATE" >= '{max(start, pd.Timestamp(f'{year}-01-01', tz='UTC')).strftime('%Y-%m-%d')}'"""
            df = fetch_paginated(rid, where, "_id", f"demand {year}", self.stdout)
            if df.empty:
                self.stdout.write(self.style.WARNING(f"    demand {year}: no rows"))
                continue
            idx = (pd.to_datetime(df["SETTLEMENT_DATE"]).dt.tz_localize("UTC")
                   + (pd.to_numeric(df["SETTLEMENT_PERIOD"]) - 1) * pd.Timedelta("30min"))
            # .to_numpy() matters: passing a Series that still carries its own
            # RangeIndex alongside index=idx makes pandas reindex it to all-NaN.
            frames.append(pd.DataFrame(
                {"demand": pd.to_numeric(df["ND"], errors="coerce").to_numpy()},
                index=pd.DatetimeIndex(idx)))
        if not frames:
            self.stderr.write("No demand data fetched; aborting.")
            return
        demand = pd.concat(frames).sort_index()
        demand = demand[~demand.index.duplicated(keep="last")]

        # ---- generation mix: solar, wind, nuclear ----
        self.stdout.write("  fetching generation mix...")
        gm = fetch_paginated(GEN_MIX_RESOURCE,
                             f""""DATETIME" >= '{start.strftime('%Y-%m-%d')}'""",
                             "_id", "gen mix", self.stdout)
        if gm.empty:
            self.stderr.write("No generation mix data; aborting.")
            return
        gidx = pd.DatetimeIndex(pd.to_datetime(gm["DATETIME"]).dt.tz_localize("UTC"))
        num = lambda c: pd.to_numeric(gm[c], errors="coerce").to_numpy()
        gen = pd.DataFrame({
            "solar": num("SOLAR"),
            "bm_wind": num("WIND"),
            "total_wind": num("WIND") + num("WIND_EMB"),
            "nuclear": num("NUCLEAR"),
        }, index=gidx).sort_index()
        gen = gen[~gen.index.duplicated(keep="last")]

        # ---- weather ----
        self.stdout.write("  fetching Open-Meteo archive...")
        wr = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
            "latitude": 54.0, "longitude": 2.3,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,wind_speed_10m,direct_radiation",
        }, timeout=300)
        wr.raise_for_status()
        h = wr.json()["hourly"]
        wx = pd.DataFrame({
            "temp_2m": h["temperature_2m"],
            "wind_10m": h["wind_speed_10m"],
            "rad": h["direct_radiation"],
        }, index=pd.to_datetime(h["time"]).tz_localize("UTC")).sort_index()
        wx = wx.resample("30min").interpolate(limit=2)
        self.stdout.write(f"    weather: {len(wx)} half-hours")

        # ---- assemble ----
        df = demand.join(gen, how="inner").join(wx, how="inner")
        df = df[(df.index >= start) & (df.index <= end)]

        gas = get_gas_ttf_history(start=start, end=end)
        if len(gas):
            df["gas_ttf"] = gas.tz_convert("UTC").reindex(df.index, method="ffill").bfill()
        else:
            df["gas_ttf"] = None

        before = len(df)
        df = df.dropna(subset=REQUIRED)
        self.stdout.write(f"  assembled {before} rows, {len(df)} complete "
                          f"({before - len(df)} dropped for missing required columns)")
        if df.empty:
            self.stderr.write("Nothing to write.")
            return

        expected = len(pd.date_range(df.index.min(), df.index.max(), freq="30min", tz="UTC"))
        self.stdout.write(f"  span {df.index.min()} -> {df.index.max()}  "
                          f"coverage {len(df)}/{expected} = {100 * len(df) / expected:.1f}%")

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("  --dry-run: nothing written"))
            self.stdout.write(df.describe().to_string())
            return

        History.objects.all().delete()
        History.objects.bulk_create([
            History(
                date_time=ts.to_pydatetime(),
                total_wind=float(r.total_wind), bm_wind=float(r.bm_wind),
                solar=float(r.solar), nuclear=float(r.nuclear or 0),
                gas_ttf=(None if pd.isna(r.gas_ttf) else float(r.gas_ttf)),
                temp_2m=float(r.temp_2m), wind_10m=float(r.wind_10m), rad=float(r.rad),
                demand=float(r.demand),
            )
            for ts, r in df.iterrows()
        ], batch_size=2000)
        self.stdout.write(self.style.SUCCESS(f"  wrote {History.objects.count()} History rows"))
