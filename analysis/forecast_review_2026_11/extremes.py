"""Were the inputs right on the extreme days, was the price inside the training range, and did others do better?"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
import django, os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()
from config.utils import day_ahead_to_agile

D = "analysis/forecast_review_2026_11/data"
pd.set_option("display.width", 250)
fd = pd.read_pickle(f"{D}/scored.pkl")
p = pd.read_csv(f"{D}/prices.csv", parse_dates=["date_time"]).set_index("date_time").sort_index()

# 1. Inputs: as forecast 2-4 d ahead vs freshest (<1 d) on the GB day
cols = ["bm_wind", "emb_wind", "solar", "demand", "dispatchable_capacity", "eu_wind_cf", "fr_wind", "gas_ttf", "nuclear", "day_ahead", "actual"]
far = fd[(fd.dt >= 2) & (fd.dt < 4)].groupby("gb_date")[cols].mean()
near = fd[(fd.dt > 0) & (fd.dt < 1)].groupby("gb_date")[cols].mean()
gw = ["bm_wind", "emb_wind", "solar", "demand", "dispatchable_capacity", "nuclear"]
cmp = pd.DataFrame({f"{c}_far": far[c] for c in cols}).join(pd.DataFrame({f"{c}_near": near[c] for c in cols}))
for c in gw:
    cmp[f"{c}_far"] /= 1000; cmp[f"{c}_near"] /= 1000
show = ["bm_wind", "demand", "dispatchable_capacity", "eu_wind_cf", "gas_ttf"]
print("== Daily mean inputs: 2-4 d forecast vs <1 d (GW, TTF EUR/MWh) ==")
print(cmp[[f"{c}_{w}" for c in show for w in ("far", "near")] + ["day_ahead_far", "actual_far"]].round(2).to_string())

# 2. Evening-peak inputs on spike days (16:00-19:30 GB) vs other Nov weekdays
fd["hr"] = fd.date_time.dt.tz_convert("GB").dt.hour
pk = fd[(fd.hr.between(16, 19)) & (fd.dt > 0) & (fd.dt < 1)]
print("\n== <1 d inputs at evening peak 16-20h (GB) by day, Nov ==")
q = pk[pk.gb_date >= pd.Timestamp("2026-11-02").date()].groupby("gb_date")[["bm_wind", "emb_wind", "demand", "dispatchable_capacity", "actual"]].agg(["mean"]).droplevel(1, axis=1)
q[["bm_wind", "emb_wind", "demand", "dispatchable_capacity"]] /= 1000
q["actual_max"] = pk[pk.gb_date >= pd.Timestamp("2026-11-02").date()].groupby("gb_date").actual.max()
print(q.round(2).to_string())

# 3. Training range: max/min settled price in the 60 d before each spike/trough day
print("\n== Training range (60 d before day) vs day extreme ==")
for d, kind in [("2026-10-24", "min"), ("2026-10-25", "min"), ("2026-11-06", "max"), ("2026-11-09", "max"), ("2026-11-10", "max")]:
    t = pd.Timestamp(d, tz="GB")
    win = p.loc[(p.index >= t - pd.Timedelta(days=60)) & (p.index < t), "day_ahead"]
    day = p.loc[(p.index >= t) & (p.index < t + pd.Timedelta(days=1)), "day_ahead"]
    print(d, kind, f"day={getattr(day, kind)():.0f}", f"train_min={win.min():.0f} train_p1={win.quantile(.01):.0f} train_p99={win.quantile(.99):.0f} train_max={win.max():.0f}",
          f"slots_beyond={(win < day.min()).sum() if kind == 'min' else (win > day.max()).sum()}")
print("\ngas TTF (stored, daily mean of <1 d rows):")
print(near.gas_ttf.iloc[::3].round(1).to_string())

# 4. External forecasters vs ours, Agile region G p/kWh, by lead bin, on tails
ext = pd.read_csv(f"{D}/external.csv")
for c in ("date_time", "source_created_at"):
    ext[c] = pd.to_datetime(ext[c], utc=True, format="mixed")
ext["dt"] = (ext.date_time - ext.source_created_at).dt.total_seconds() / 86400
act = day_ahead_to_agile(p.day_ahead, region="G")
ours = fd[["date_time", "dt", "day_ahead", "forecast_id"]].copy()
ours["agile_pred"] = np.nan
for fid, g in ours.groupby("forecast_id"):
    ours.loc[g.index, "agile_pred"] = day_ahead_to_agile(g.set_index("date_time").day_ahead, region="G").values
ours["source"] = "agilepredict"
both = pd.concat([ext[["source", "date_time", "dt", "agile_pred"]], ours[["source", "date_time", "dt", "agile_pred"]]])
both = both[(both.date_time >= "2026-10-14") & (both.dt > 0)]
both["actual"] = both.date_time.map(act)
both = both.dropna(subset=["actual"])
lo, hi = act.loc["2026-10-14":].quantile([0.1, 0.9])
both["tail"] = np.select([both.actual <= lo, both.actual >= hi], ["low10", "high10"], "mid")
both["err"] = both.agile_pred - both.actual
both["hbin"] = pd.cut(both.dt, [0, 1, 2, 4, 7])
both.to_pickle(f"{D}/vs_external.pkl")
# only slots every source covered at that lead bin, so the comparison is like-for-like
cov = both.groupby(["hbin", "date_time"], observed=True).source.nunique()
full = cov[cov == both.source.nunique()].index
both = both.set_index(["hbin", "date_time"]).loc[lambda x: x.index.isin(full)].reset_index()
print(f"\n== Agile G p/kWh, common slots only: p10={lo:.1f} p90={hi:.1f} ==")
r = both.groupby(["hbin", "tail", "source"], observed=True).err.agg(n="size", bias="mean", mae=lambda e: e.abs().mean())
print(r.round(2).unstack("source").to_string())
