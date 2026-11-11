"""E2: does NESO's 2-day-ahead margin forecast (OPMR National Surplus) flag the price spikes?

Daily frame, Jul 2023 -> now: daily max price over (TTF + 12) [12 = fitted fixed offset, stack.py]
against the surplus NESO published 2 days before, and against actual peak residual demand.
Run: .venv/bin/python analysis/forecast_review_2026_11/margin.py
"""
import numpy as np
import pandas as pd

D = "analysis/forecast_review_2026_11/data"
p = pd.read_pickle(f"{D}/panel.pkl")
o = pd.read_pickle(f"{D}/opmr.pkl")
p["gbdate"] = p.index.tz_convert("Europe/London").tz_localize(None).normalize()
d = p.groupby("gbdate").agg(pmax=("day_ahead", "max"), pmean=("day_ahead", "mean"), pmin=("day_ahead", "min"),
                            ttf=("gas_ttf", "mean"), rmax=("resid", "max"), rmin=("resid", "min"), n=("day_ahead", "size"))
d = d[d.n >= 46].join(o)
d["nmax"] = d.pmax / (d.ttf + 12)   # gas-normalised daily peak
d["nmean"] = d.pmean / (d.ttf + 12)
d["prior60max"] = d.pmax.shift(1).rolling(60).max()
d["breach"] = d.pmax > d.prior60max
m, y = d.index.month, d.index.year
d["season"] = np.where(m >= 10, y, np.where(m <= 3, y - 1, 0))
d["wkend"] = d.index.dayofweek >= 5
d.to_pickle(f"{D}/daily.pkl")
w = d[d.season > 0]

pd.set_option("display.width", 250)
print("Spearman rank correlation with the gas-normalised daily peak, winter days")
for s, g in w.groupby("season"):
    print(f"  W{s}: n={len(g)}  surplus_L2 {g.nmax.corr(g.surplus_L2, method='spearman'):+.2f}  surplus_L4 {g.nmax.corr(g.surplus_L4, method='spearman'):+.2f}"
          f"  surplus_L7 {g.nmax.corr(g.surplus_L7, method='spearman'):+.2f}  actual peak resid {g.nmax.corr(g.rmax, method='spearman'):+.2f}"
          f"  disp_L2 (app feature) {g.nmax.corr(g.disp_L2, method='spearman'):+.2f}")

w = w.assign(sbin=pd.cut(w.surplus_L2 / 1000, [-10, 2, 4, 6, 8, 12, 30]))
print("\nwinter days by 2-day-ahead surplus (GW): count / median and 90th pct of normalised daily peak / share breaching the prior 60-day max")
t = w.groupby(["sbin"], observed=False).agg(days=("nmax", "size"), med=("nmax", "median"), p90=("nmax", lambda s: s.quantile(0.9)), breach=("breach", "mean"), med_price=("pmax", "median"))
print(t.round(2).to_string())
print("\nsame, median normalised peak by season")
print(w.pivot_table(index="season", columns="sbin", values="nmax", aggfunc="median", observed=False).round(2).to_string())
print(w.pivot_table(index="season", columns="sbin", values="nmax", aggfunc="size", observed=False).to_string())

# Spike days: gas-normalised daily peak in the top 5% of all days. How well does each signal rank them?
from sklearn.metrics import roc_auc_score
a = d.dropna(subset=["surplus_L2", "nmax"])
spike = a.nmax >= a.nmax.quantile(0.95)
print(f"\nspike days (top 5% normalised peak, >= {a.nmax.quantile(0.95):.2f}): {spike.sum()} of {len(a)}")
for c in ["surplus_L2", "surplus_L3", "surplus_L4", "surplus_L7", "disp_L2"]:
    print(f"  AUC {c}: {roc_auc_score(spike, -a[c].fillna(a[c].median())):.3f}")
print(f"  AUC actual peak residual demand: {roc_auc_score(spike, a.rmax):.3f}")
x = (a.rmax - a.disp_L2)
print(f"  AUC actual peak residual demand minus disp_L2: {roc_auc_score(spike, x):.3f}")
print("\nspike days by month:", spike[spike].groupby(spike[spike].index.to_period("M")).size().to_dict())
print("\nNov 2026")
print(d.loc["2026-10-26":, ["pmax", "pmean", "ttf", "nmax", "surplus_L2", "surplus_L4", "rmax", "prior60max", "breach"]].round(2).to_string())
