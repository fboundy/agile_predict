"""NESO OPMR (2-14 day ahead daily margin forecast) history, joined to the panel at fixed lead times.

Writes data/opmr.pkl: one row per target date with the values published 2, 3, 4 and 7 days before.
Run: .venv/bin/python analysis/forecast_review_2026_11/opmr_hist.py
"""
import pandas as pd
import requests

D = "analysis/forecast_review_2026_11/data"
RID = "0eede912-8820-4c66-a58a-f7436d36b95f"
recs = []
for a, b in [("2023-06-01", "2024-09-01"), ("2024-09-01", "2025-12-01"), ("2025-12-01", "2027-01-01")]:
    sql = f'SELECT * FROM "{RID}" WHERE "Publish Date" >= \'{a}\' AND "Publish Date" < \'{b}\' ORDER BY "_id" LIMIT 20000'
    r = requests.get("https://api.neso.energy/api/3/action/datastore_search_sql", params={"sql": sql}, timeout=120).json()["result"]["records"]
    print(a, len(r))
    recs += r
o = pd.DataFrame(recs).drop(columns=["_full_text", "_id"])
o["Date"] = pd.to_datetime(o["Date"])
o["Publish Date"] = pd.to_datetime(o["Publish Date"])
num = [c for c in o.columns if c not in ("Date", "Publish Date")]
o[num] = o[num].apply(pd.to_numeric, errors="coerce")
o["lead"] = (o["Date"] - o["Publish Date"]).dt.days
o = o.rename(columns={"Generator Availability": "avail", "Maximum IC Import": "ic", "OPMR total": "opmr", "Constrained Plant": "constr",
                      "National Surplus": "surplus", "Peak Demand Forecast": "peak_dem", "Minimum Demand Forecast": "min_dem",
                      "Maximum IC Export": "ic_export", "Negative Reserve": "neg_res"})
o["constr"] = o["constr"].fillna(0)
o["disp"] = o.avail + o.ic - o.opmr - o.constr  # the app's dispatchable_capacity
print("lead-time counts:", o.lead.value_counts().sort_index().to_dict())
keep = ["avail", "ic", "ic_export", "opmr", "constr", "surplus", "peak_dem", "min_dem", "neg_res", "disp"]
out = pd.concat({f"L{k}": o[o.lead == k].drop_duplicates("Date", keep="last").set_index("Date")[keep] for k in (2, 3, 4, 7)}, axis=1)
out.columns = [f"{c}_{l}" for l, c in out.columns]
out.to_pickle(f"{D}/opmr.pkl")
print(out.shape, out.index.min(), out.index.max(), "missing L2 days:", out["surplus_L2"].isna().sum())
print(out[["surplus_L2", "avail_L2", "ic_L2", "peak_dem_L2", "disp_L2"]].describe().round(0).to_string())
print(out.loc["2026-10-20":"2026-11-12", ["surplus_L2", "surplus_L4", "surplus_L7", "avail_L2", "peak_dem_L2", "disp_L2"]].to_string())
