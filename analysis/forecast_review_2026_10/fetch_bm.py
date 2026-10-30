"""Fetch wind PN + BOALF from Elexon and derive half-hourly wind curtailment by region.

Curtailment = max(PN - accepted BOA level, 0), sampled at 5-minute resolution where a
BOA instruction covers the instant (latest acceptance wins). Region from the unit's
transmission loss factor (zonal): -0.0305 North Scotland, -0.0099 South Scotland.
Units without a TLF are kept in an 'unclassified' bucket.
"""
import sys

import numpy as np
import pandas as pd
import requests

B = "https://data.elexon.co.uk/bmrs/api/v1"
D = "analysis/forecast_review_2026_10/data"
TAG = sys.argv[3] if len(sys.argv) > 3 else ""
START, END = sys.argv[1], sys.argv[2]  # e.g. 2026-10-12 2026-10-31

units = pd.DataFrame(requests.get(f"{B}/reference/bmunits/all", timeout=60).json())
units = units[units.fuelType == "WIND"].copy()
tlf = pd.to_numeric(units.transmissionLossFactor, errors="coerce")
units["region"] = np.select(
    [tlf.round(4) == -0.0305, tlf.round(4) == -0.0099, tlf.notna()], ["scotland", "scotland", "england_wales"], "unclassified"
)
ids = units.elexonBmUnit.dropna().tolist()
region = dict(zip(units.nationalGridBmUnit, units.region))


def stream(ds, day):
    out = []
    for i in range(0, len(ids), 80):
        r = requests.get(f"{B}/datasets/{ds}/stream", params={"from": f"{day}T00:00Z", "to": f"{day}T23:59Z", "bmUnit": ids[i:i + 80]}, timeout=120)
        r.raise_for_status()
        out += r.json()
    return out


def interp(g, rows):
    """Piecewise-linear level on the grid from segment rows; NaN outside all segments."""
    v = np.full(len(g), np.nan)
    for tf, tt, lf, lt in rows:
        a, b = pd.Timestamp(tf).value, pd.Timestamp(tt).value
        m = (g >= a) & (g <= b)
        if m.any():
            v[m] = lf + (lt - lf) * (g[m] - a) / max(b - a, 1)
    return v


def day_curtailment(d):
    """Half-hourly PN and curtailment by region for one day (processed per day to fit the CT's 1 GB RAM)."""
    pn, boa = pd.DataFrame(stream("PN", d)), pd.DataFrame(stream("BOALF", d))
    grid = pd.date_range(f"{d}T00:00Z", f"{d}T23:55Z", freq="5min")
    g = grid.values.astype("int64")
    sums = {f"{k}_{r}": np.zeros(len(g)) for k in ("pn", "curt") for r in ("scotland", "england_wales", "unclassified")}
    cols = ["timeFrom", "timeTo", "levelFrom", "levelTo"]
    for unit, p in pn.groupby("nationalGridBmUnit"):
        pv = interp(g, p.sort_values("timeFrom")[cols].itertuples(index=False))
        bv = np.full(len(g), np.nan)
        if len(boa):
            b = boa[boa.nationalGridBmUnit == unit]
            bv = interp(g, b.sort_values(["acceptanceTime", "acceptanceNumber"])[cols].itertuples(index=False))
        reg = region.get(unit, "unclassified")
        sums[f"pn_{reg}"] += np.nan_to_num(pv)
        sums[f"curt_{reg}"] += np.where(np.isnan(bv), 0, np.clip(np.nan_to_num(pv) - bv, 0, None))
    return pd.DataFrame(sums, index=grid).resample("30min").mean()


units[["nationalGridBmUnit", "elexonBmUnit", "region", "generationCapacity", "transmissionLossFactor"]].to_csv(f"{D}/wind_units.csv", index=False)
parts = []
for day in pd.date_range(START, END):
    parts.append(day_curtailment(day.strftime("%Y-%m-%d")))
    print(day.date(), flush=True)
out = pd.concat(parts)
out.to_csv(f"{D}/wind_curtailment_hh{TAG}.csv")
print(out.resample("1D").mean().div(1000).round(2).to_string())
