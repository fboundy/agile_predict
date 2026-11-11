"""E7: the cheapest candidate, scored on the forecasts we actually issued.

Fit, on winter days BEFORE 14 Oct 2026 only, the typical gas-normalised daily peak for a given NESO
surplus (isotonic, decreasing). Then for each stored 2-4 d forecast since 14 Oct, lift the evening
(16-19) slots so the day's peak is at least that figure for the surplus stored with the forecast.
No retraining, real forecast-time inputs, so this is out of sample and not an offline replay.
Run: .venv/bin/python analysis/forecast_review_2026_11/uplift.py   (needs data/h24.pkl, data/daily.pkl)
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

D = "analysis/forecast_review_2026_11/data"
d = pd.read_pickle(f"{D}/daily.pkl")
tr = d[(d.season > 0) & (d.index < "2026-10-14")].dropna(subset=["surplus_L2", "nmax"])
h = pd.read_pickle(f"{D}/h24.pkl").dropna(subset=["opmr_national_surplus", "gas_ttf"])
pd.set_option("display.width", 250)

def score(q, label):
    iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
    if q is None:
        iso.fit(tr.surplus_L2, tr.nmax)  # conditional mean
    else:  # conditional quantile: isotonic fit to the rolling quantile over surplus-sorted days
        s = tr.sort_values("surplus_L2")
        iso.fit(s.surplus_L2, s.nmax.rolling(41, center=True, min_periods=15).quantile(q))
    g = h.groupby(["forecast_id", "gbdate"])
    day = g.agg(n=("actual", "size"), fpk=("day_ahead", "max"), apk=("actual", "max"), sur=("opmr_national_surplus", "min"), ttf=("gas_ttf", "mean"))
    day["floor"] = iso.predict(day.sur) * (day.ttf + 12)
    x = h.join(day[["fpk", "floor"]], on=["forecast_id", "gbdate"])
    k = np.where(x.hour.between(16, 18), np.maximum(1, x.floor / x.fpk), 1.0)
    x["new"] = x.day_ahead * k
    day = day[day.n >= 40].join(x.groupby(["forecast_id", "gbdate"]).new.max().rename("npk"))
    tight = day.sur < 6000
    lifted = day.npk > day.fpk + 1
    e0, e1 = x.day_ahead - x.actual, x.new - x.actual
    ev = x.hour.between(16, 18)
    print(f"{label}: all-slot RMSE {np.sqrt((e0 ** 2).mean()):.1f} -> {np.sqrt((e1 ** 2).mean()):.1f}; evening RMSE {np.sqrt((e0[ev] ** 2).mean()):.1f} -> {np.sqrt((e1[ev] ** 2).mean()):.1f}; "
          f"evening bias {e0[ev].mean():+.1f} -> {e1[ev].mean():+.1f}")
    for lab, m in (("surplus < 6 GW", tight), ("the rest", ~tight)):
        a = day[m]
        print(f"   {lab}: run-days {len(a)}, lifted {lifted[m].mean():.0%}; peak error {(a.fpk - a.apk).mean():+.0f} -> {(a.npk - a.apk).mean():+.0f}; "
              f"peak MAE {(a.fpk - a.apk).abs().mean():.0f} -> {(a.npk - a.apk).abs().mean():.0f}")
    return day

print(f"fitted on {len(tr)} winter days, {tr.index.min():%b %Y} - {tr.index.max():%d %b %Y}")
for q, label in ((None, "floor = mean peak for the surplus"), (0.5, "floor = median"), (0.75, "floor = 75th pct")):
    day = score(q, label)
t = day.groupby("gbdate").agg(sur=("sur", "mean"), actual=("apk", "mean"), forecast=("fpk", "mean"), floor=("floor", "mean"))
print("\nper target day, 75th-pct floor (mean over 2-4 d runs), days with surplus < 8 GW")
print(t[t.sur < 8000].round(0).to_string())
