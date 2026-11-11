"""Fill panel.pkl's gas_ttf. Yahoo's TTF=F history endpoint returned only the latest quote on
2026-11-11 (any range), so stitch: History table (to 6 Aug 2026), the October review's
drivers file (Jun - 30 Oct, fetched from Yahoo when it worked), then each stored run's value.
Run: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/panel_gas.py
"""
import pandas as pd
from prices.models import History

D = "analysis/forecast_review_2026_11/data"
p = pd.read_pickle(f"{D}/panel.pkl")
old = pd.DataFrame(History.objects.values("date_time", "gas_ttf")).set_index("date_time")["gas_ttf"].sort_index()
oct_ = pd.read_csv("analysis/forecast_review_2026_10/data/drivers_hh.csv", index_col=0, parse_dates=True)["gas_ttf"].dropna()
fd = pd.read_csv(f"{D}/forecast_data.csv", usecols=["forecast_id", "gas_ttf"]).groupby("forecast_id")["gas_ttf"].first()
ff = pd.read_csv(f"{D}/forecasts.csv", index_col="id")
runs = pd.Series(fd.values, index=pd.to_datetime(ff.loc[fd.index, "created_at"], utc=True, format="mixed")).sort_index()
runs = runs[runs.index > oct_.index.max()]
gas = pd.concat([old[old.index < oct_.index.min()], oct_, runs]).sort_index()
both = pd.concat([old, oct_], axis=1, keys=["hist", "oct"]).dropna()
print("History vs Oct-file overlap rows", len(both), "max abs diff", round((both["hist"] - both["oct"]).abs().max(), 3))
p["gas_ttf"] = gas[~gas.index.duplicated()].reindex(p.index, method="ffill").bfill()
assert p["gas_ttf"].notna().all()
p.to_pickle(f"{D}/panel.pkl")
print(p["gas_ttf"].resample("QS").agg(["mean", "min", "max"]).round(1).to_string())
