"""Extend panel.pkl past the NESO generation mix's ~1 week lag using Elexon FUELHH (transmission
metered) for the dispatchable side, and the stored <1 d forecast for embedded wind and solar.
Checks the FUELHH-built residual demand against the generation mix on the overlap first.
Run: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/panel_tail.py
"""
import pandas as pd
import requests
from prices.models import PriceHistory

D = "analysis/forecast_review_2026_11/data"
p = pd.read_pickle(f"{D}/panel.pkl")
p = p[p.index <= "2026-11-04 23:30+00:00"]  # idempotent: drop any earlier extension
rows = []
for d in pd.date_range("2026-10-15", pd.Timestamp.now().normalize(), freq="7D"):
    e = d + pd.Timedelta(days=6)
    rows += requests.get("https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/stream",
                         params={"settlementDateFrom": f"{d:%Y-%m-%d}", "settlementDateTo": f"{e:%Y-%m-%d}"}, timeout=120).json()
f = pd.DataFrame(rows)
f["startTime"] = pd.to_datetime(f.startTime, utc=True)
f = f.pivot_table(index="startTime", columns="fuelType", values="generation", aggfunc="mean")
ints = [c for c in f.columns if c.startswith("INT")]
print("fuel types:", f.columns.tolist())
t = pd.DataFrame({
    "gas": f[["CCGT", "OCGT"]].sum(axis=1), "coal": f.get("COAL", 0), "nuclear": f["NUCLEAR"],
    "hydro": f["NPSHYD"], "imports": f[ints].clip(lower=0).sum(axis=1), "biomass": f["BIOMASS"],
    "other": f["OTHER"], "storage": f["PS"].clip(lower=0), "wind_tx": f["WIND"],
})
t["resid"] = t[["gas", "coal", "hydro", "imports", "biomass", "other", "storage"]].sum(axis=1)
ov = t.join(p[["resid", "nuclear", "gas", "imports"]], rsuffix="_mix", how="inner")
print("overlap rows", len(ov), "resid corr", round(ov.resid.corr(ov.resid_mix), 4),
      "mean diff", round((ov.resid - ov.resid_mix).mean()), "rmse", round(((ov.resid - ov.resid_mix) ** 2).mean() ** 0.5),
      "| gas diff", round((ov.gas - ov.gas_mix).mean()), "imports diff", round((ov.imports - ov.imports_mix).mean()))

# embedded wind and solar after 4 Nov: latest stored forecast for each slot
fd = pd.read_csv(f"{D}/forecast_data.csv", usecols=["forecast_id", "date_time", "emb_wind", "solar", "eu_wind_cf", "gas_ttf"], parse_dates=["date_time"])
fd = fd.sort_values("forecast_id").drop_duplicates("date_time", keep="last").set_index("date_time")
new = t[t.index > p.index.max()].join(fd, how="inner")
price = pd.DataFrame(PriceHistory.objects.filter(date_time__gt=p.index.max()).values("date_time", "day_ahead")).set_index("date_time")["day_ahead"]
new = new.join(price, how="inner")
new["wind"] = new["wind_tx"] + new["emb_wind"]
new["wind_emb"] = new["emb_wind"]
new["generation"] = new["resid"] + new["wind"] + new["solar"] + new["nuclear"]
p = pd.concat([p, new[p.columns]]).sort_index()
p.to_pickle(f"{D}/panel.pkl")
print(p.shape, p.index.max())
print(p["2026-11-05":].resample("D").agg({"day_ahead": ["mean", "max"], "resid": "max", "wind": "min", "gas_ttf": "mean"}).round(0).to_string())
