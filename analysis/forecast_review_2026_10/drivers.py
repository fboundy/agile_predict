"""Assemble half-hourly ACTUAL drivers, Jun-Oct 2026, for the structural price test.

Run: .venv/bin/python manage.py shell < analysis/forecast_review_2026_10/drivers.py
"""
import numpy as np
import pandas as pd
import requests

from config.utils import get_gas_ttf_history
from prices.models import PriceHistory

D = "analysis/forecast_review_2026_10/data"
START, END = "2026-06-01", "2026-10-30"
B = "https://data.elexon.co.uk/bmrs/api/v1"
idx = pd.date_range(f"{START}T00:00Z", f"{END}T23:30Z", freq="30min")

price = pd.DataFrame(PriceHistory.objects.filter(date_time__gte=idx[0], date_time__lte=idx[-1]).values("date_time", "day_ahead"))
price = price.set_index("date_time")["day_ahead"]

# FUELHH: transmission demand proxy (all generation + net imports) and net imports
rows = []
for d in pd.date_range(START, END, freq="7D"):
    e = min(d + pd.Timedelta(days=6), pd.Timestamp(END))
    rows += requests.get(f"{B}/datasets/FUELHH/stream", params={"settlementDateFrom": d.strftime("%Y-%m-%d"), "settlementDateTo": e.strftime("%Y-%m-%d")}, timeout=120).json()
f = pd.DataFrame(rows)
f["startTime"] = pd.to_datetime(f.startTime)
f = f.pivot_table(index="startTime", columns="fuelType", values="generation", aggfunc="mean")
ints = [c for c in f.columns if c.startswith("INT")]
fuel = pd.DataFrame({
    "tx_demand": f.sum(axis=1),
    "ic_net_import": f[ints].sum(axis=1),
    "metered_wind": f["WIND"],
    "nuclear": f["NUCLEAR"],
})

# B1630 actual solar (includes embedded) and onshore/offshore wind
rows = []
for d in pd.date_range(START, END, freq="7D"):
    e = min(d + pd.Timedelta(days=7), pd.Timestamp(END) + pd.Timedelta(days=1))
    rows += requests.get(f"{B}/generation/actual/per-type/wind-and-solar", params={"from": d.strftime("%Y-%m-%dT00:00Z"), "to": e.strftime("%Y-%m-%dT00:00Z")}, timeout=120).json()["data"]
a = pd.DataFrame(rows)
a["startTime"] = pd.to_datetime(a.startTime)
a = a.pivot_table(index="startTime", columns="psrType", values="quantity", aggfunc="mean")
print("B1630 types:", a.columns.tolist())

# Gas TTF (daily close, forward-filled)
gas = get_gas_ttf_history(START, END)
gas.index = gas.index.tz_convert("UTC").normalize()
gas = gas.groupby(level=0).last()

# Continental wind proxy: Open-Meteo historical forecasts, 100 m wind, NW-Europe wind belt
PTS = {"de_north": (53.6, 9.3), "de_schleswig": (54.6, 9.2), "de_bight": (54.4, 6.6), "nl": (52.9, 5.6),
       "dk": (56.0, 8.6), "be_coast": (51.5, 2.8), "fr_north": (50.2, 2.8), "fr_central": (47.0, 2.0)}
r = requests.get("https://historical-forecast-api.open-meteo.com/v1/forecast", params={
    "latitude": ",".join(str(v[0]) for v in PTS.values()), "longitude": ",".join(str(v[1]) for v in PTS.values()),
    "hourly": "wind_speed_100m,wind_speed_10m", "wind_speed_unit": "ms", "start_date": START, "end_date": END, "timezone": "UTC"}, timeout=120).json()
wx = {}
for name, loc in zip(PTS, r):
    t = pd.to_datetime(loc["hourly"]["time"]).tz_localize("UTC")
    wx[f"{name}_ws100"] = pd.Series(loc["hourly"]["wind_speed_100m"], index=t)
    wx[f"{name}_ws10"] = pd.Series(loc["hourly"]["wind_speed_10m"], index=t)
wx = pd.DataFrame(wx).resample("30min").interpolate()


def cf(v):
    """Crude wind-turbine power curve: 0 below 3 m/s, cubic to rated at 12 m/s, 0 above 25."""
    return np.where(v < 3, 0, np.where(v >= 25, 0, np.clip((v ** 3 - 27) / (12 ** 3 - 27), 0, 1)))


belt = [k for k in PTS if k != "fr_central"]
wx["eu_wind_cf"] = np.mean([cf(wx[f"{k}_ws100"]) for k in belt], axis=0)

out = pd.DataFrame(index=idx)
out["price"] = price
out = out.join(fuel).join(a.add_prefix("b1630_")).join(wx)
out["gas_ttf"] = gas.reindex(idx, method="ffill")
out.to_csv(f"{D}/drivers_hh.csv")
print(out.describe().T.round(1).to_string())
