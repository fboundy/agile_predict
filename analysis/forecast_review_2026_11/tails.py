"""E5: (a) how often does the day's price leave the range of a 60-day training window?
(b) the low tail: what marks the zero/negative-price days?
Run: .venv/bin/python analysis/forecast_review_2026_11/tails.py   (needs data/daily.pkl from margin.py)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

D = "analysis/forecast_review_2026_11/data"
d = pd.read_pickle(f"{D}/daily.pkl")
p = pd.read_pickle(f"{D}/panel.pkl")
p["gbdate"] = p.index.tz_convert("Europe/London").tz_localize(None).normalize()
d = d.join(p.groupby("gbdate").agg(eu=("eu_wind_cf", "mean"), wind=("wind", "mean"), solar=("solar", "max"), nuc=("nuclear", "mean")))
d["nmin"] = d.pmin / (d.ttf + 12)
pd.set_option("display.width", 250)

print("== (a) days outside the range seen in the prior N days (the model trains on 60) ==")
rows = []
for n in (60, 365, 730):
    for lab, hi, lo in (("raw", d.pmax, d.pmin), ("gas-normalised", d.nmax, d.nmin)):
        up = hi > hi.shift(1).rolling(n).max()
        dn = lo < lo.shift(1).rolling(n).min()
        ok = hi.shift(1).rolling(n).count() == n
        rows.append(dict(window=n, target=lab, days=ok.sum(), above=up[ok].sum(), below=dn[ok].sum(),
                         above_winter=up[ok & (d.season > 0)].sum(), med_excess=(hi / hi.shift(1).rolling(n).max() - 1)[up & ok].median()))
print(pd.DataFrame(rows).round(2).to_string(index=False))
up60 = d.pmax > d.prior60max
spike = d.nmax >= d.nmax.quantile(0.95)
print(f"spike days (top 5% normalised peak): {spike.sum()}; of which above the prior 60-day max: {(spike & up60).sum()}; above the prior 365-day max: {(spike & (d.pmax > d.pmax.shift(1).rolling(365).max())).sum()}")
print("days above prior 60-day max, by month:", up60[up60].groupby(up60[up60].index.month).size().to_dict())
r = d.pmax / d.prior60max
print("peak / prior 60-day max on spike days: quartiles", r[spike].quantile([.25, .5, .75]).round(2).tolist())

print("\n== (b) low tail: days with a zero or negative slot ==")
neg = d.pmin <= 0
print(f"days with min <= 0: {neg.sum()} of {len(d)}; by year: {neg[neg].groupby(neg[neg].index.year).size().to_dict()}; by month: {neg[neg].groupby(neg[neg].index.month).size().to_dict()}")
for c, sign in (("rmin", -1), ("wind", 1), ("eu", 1), ("solar", 1), ("neg_res_L2", -1), ("min_dem_L2", -1), ("ic_export_L2", -1), ("surplus_L2", 1), ("wkend", 1)):
    a = d.dropna(subset=[c])
    print(f"  AUC {c:14s} {roc_auc_score(a.pmin <= 0, sign * a[c].astype(float)):.3f}  (n={len(a)})")
a = d.dropna(subset=["eu", "rmin"])
a = a.assign(rb=pd.cut(a.rmin / 1000, [-5, 5, 8, 11, 15, 40]), eb=pd.qcut(a.eu, 3, labels=["low", "mid", "high"]))
print("\nshare of days with a slot <= 0, by the day's minimum residual demand (GW) and continental wind tercile")
print(a.pivot_table(index="rb", columns="eb", values="pmin", aggfunc=lambda s: (s <= 0).mean(), observed=False).round(2).to_string())
print(a.pivot_table(index="rb", columns="eb", values="pmin", aggfunc="size", observed=False).to_string())
print("\nmedian daily minimum price (GBP/MWh), same cells")
print(a.pivot_table(index="rb", columns="eb", values="pmin", aggfunc="median", observed=False).round(0).to_string())

h = pd.read_pickle(f"{D}/h24.pkl")
print("\n== (c) stored 2-4 d forecasts on low slots (actual <= 20) since 14 Oct ==")
low = h.actual <= 20   # all of these fall before 30 Oct, i.e. under the production feature set
print(f"  slots {len(h)}, low {low.sum()} on {h.gbdate[low].nunique()} days; mean forecast on low slots {h.day_ahead[low].mean():.0f} (actual {h.actual[low].mean():.0f}); "
      f"forecast <= 40 on {(h.day_ahead[low] <= 40).mean():.0%} of them; of forecasts <= 40, actual <= 20 on {(h.actual[h.day_ahead <= 40] <= 20).mean():.0%}")
print(f"  AUC for a low slot: the forecast {roc_auc_score(low, -h.day_ahead):.3f}; forecast-input residual demand {roc_auc_score(low, -h.resid_fc):.3f}; eu_wind_cf {roc_auc_score(low, h.eu_wind_cf):.3f}")
lowday = h.groupby("gbdate").agg(amin=("actual", "min"), fmin=("day_ahead", "min"), rmin=("resid_fc", "min"), eu=("eu_wind_cf", "mean"))
print(lowday[lowday.amin <= 20].round(1).to_string())
