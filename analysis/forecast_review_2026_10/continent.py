"""Daily GB interconnector net flow (FUELHH) alongside continental day-ahead prices (energy-charts)."""
import pandas as pd
import requests

D = "analysis/forecast_review_2026_10/data"
f = pd.read_csv(f"{D}/fuelhh.csv", index_col=0, parse_dates=True)
ic = f.filter(like="INT").sum(axis=1)  # +ve = import to GB
out = pd.DataFrame({"gb_net_import_GW": ic.resample("1D").mean() / 1000})
import time
for bzn in ["DE-LU", "FR", "NL", "NO2"]:
    time.sleep(15)
    r = requests.get("https://api.energy-charts.info/price", params={"bzn": bzn, "start": "2026-10-12", "end": "2026-10-31"}, timeout=60).json()
    s = pd.Series(r["price"], index=pd.to_datetime(r["unix_seconds"], unit="s", utc=True))
    out[bzn + "_eur"] = s.resample("1D").mean()
p = pd.read_csv(f"{D}/prices.csv", parse_dates=["date_time"]).set_index("date_time")
out["GB_gbp"] = p.day_ahead.resample("1D").mean()
out.index = out.index.date
print(out.round(1).to_string())
