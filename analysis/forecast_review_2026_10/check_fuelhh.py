"""Validate derived curtailment: metered transmission wind (FUELHH) vs PN - curtailment."""
import pandas as pd
import requests

D = "analysis/forecast_review_2026_10/data"
rows = []
for d in pd.date_range("2026-10-12", "2026-10-30", freq="7D"):
    r = requests.get("https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/stream",
                     params={"settlementDateFrom": d.strftime("%Y-%m-%d"), "settlementDateTo": min(d + pd.Timedelta(days=6), pd.Timestamp("2026-10-30")).strftime("%Y-%m-%d")}, timeout=120)
    rows += r.json()
f = pd.DataFrame(rows)
f["startTime"] = pd.to_datetime(f.startTime)
f = f.pivot_table(index="startTime", columns="fuelType", values="generation", aggfunc="mean")
f.to_csv(f"{D}/fuelhh.csv")
w = pd.read_csv(f"{D}/wind_curtailment_hh.csv", index_col=0, parse_dates=True)
w["pn_total"] = w.filter(like="pn_").sum(axis=1)
w["curt_total"] = w.filter(like="curt_").sum(axis=1)
j = w.join(f[["WIND"]], how="inner")
j["implied"] = j.pn_total - j.curt_total
print(j[["pn_total", "curt_total", "implied", "WIND"]].resample("1D").mean().div(1000).round(2).to_string())
print("corr implied vs metered:", j.implied.corr(j.WIND).round(3), " corr PN vs metered:", j.pn_total.corr(j.WIND).round(3))
print(f.columns.tolist())
