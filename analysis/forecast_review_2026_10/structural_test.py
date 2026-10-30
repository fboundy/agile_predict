"""Which information explains GB day-ahead price? LightGBM on ACTUAL half-hourly drivers, Jun-Oct 2026.

Two evaluations:
  lowo  - leave-one-week-out: does the information explain price at all?
  fwd   - production-like: train on prior 60 days, predict the next 7, rolling from 1 Aug.
Tails = actual price in the bottom / top decile of the evaluated rows.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

D = "analysis/forecast_review_2026_10/data"
x = pd.read_csv(f"{D}/drivers_hh.csv", index_col=0, parse_dates=True)
w = pd.read_csv(f"{D}/wind_curtailment_hh_hist.csv", index_col=0, parse_dates=True)
x = x.join(w)
x["pn_wind"] = x.filter(like="pn_").sum(axis=1)
x["curt_scot"] = x["curt_scotland"]
x["scot_share"] = x["pn_scotland"] / x["pn_wind"].clip(lower=500)
x["solar"] = x["b1630_Solar"]
x["emb_onshore"] = x["b1630_Wind Onshore"]
x["resid_pn"] = x["tx_demand"] - x["pn_wind"] - x["nuclear"]
gb_t = x.index.tz_convert("GB")
x["time"] = gb_t.hour + gb_t.minute / 60
x["weekend"] = (gb_t.dayofweek >= 5).astype(int)
x = x.dropna(subset=["price", "tx_demand", "pn_wind", "gas_ttf", "eu_wind_cf", "solar"])

BASE = ["tx_demand", "solar", "nuclear", "time", "weekend", "gas_ttf"]
SETS = {
    "gb_pn":          BASE + ["pn_wind"],
    "gb_metered":     BASE + ["metered_wind"],
    "gb_pn+curt":     BASE + ["pn_wind", "curt_scot"],
    "gb_pn+scotshare": BASE + ["pn_wind", "scot_share"],
    "gb_pn+fr_pt":    BASE + ["pn_wind", "fr_central_ws10"],
    "gb_pn+eu_wind":  BASE + ["pn_wind", "eu_wind_cf"],
    "gb_pn+eu+scot":  BASE + ["pn_wind", "eu_wind_cf", "scot_share", "curt_scot"],
}
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20, verbose=-1, random_state=0, n_jobs=1)


def score(y, p):
    lo, hi = y <= y.quantile(0.1), y >= y.quantile(0.9)
    return {"rmse": np.sqrt(((p - y) ** 2).mean()), "rmse_lo10": np.sqrt(((p - y)[lo] ** 2).mean()),
            "bias_lo10": (p - y)[lo].mean(), "rmse_hi10": np.sqrt(((p - y)[hi] ** 2).mean()), "bias_hi10": (p - y)[hi].mean(),
            "sd_ratio": p.std() / y.std()}


def lowo(feats, target_ratio=False):
    pred = pd.Series(index=x.index, dtype=float)
    wk = x.index.to_period("W")
    for k in wk.unique():
        tr, te = wk != k, wk == k
        y = x.price / x.gas_ttf if target_ratio else x.price
        m = LGBMRegressor(**P).fit(x.loc[tr, feats], y[tr])
        pred[te] = m.predict(x.loc[te, feats]) * (x.gas_ttf[te] if target_ratio else 1)
    return pred


def fwd(feats, target_ratio=False):
    pred = pd.Series(index=x.index, dtype=float)
    for s in pd.date_range("2026-08-01", "2026-10-30", freq="7D", tz="UTC"):
        tr = (x.index >= s - pd.Timedelta(days=60)) & (x.index < s)
        te = (x.index >= s) & (x.index < s + pd.Timedelta(days=7))
        if not te.any():
            continue
        y = x.price / x.gas_ttf if target_ratio else x.price
        m = LGBMRegressor(**P).fit(x.loc[tr, feats], y[tr])
        pred[te] = m.predict(x.loc[te, feats]) * (x.gas_ttf[te] if target_ratio else 1)
    return pred


rows, review = [], {}
for name, feats in SETS.items():
    for mode, fn in (("lowo", lowo), ("fwd", fwd)):
        for ratio in (False, True):
            if ratio and name not in ("gb_pn", "gb_pn+eu_wind"):
                continue
            p = fn(feats, ratio)
            ok = p.notna()
            label = name + (" [price/gas]" if ratio else "")
            rows.append({"set": label, "eval": mode, **score(x.price[ok], p[ok])})
            if mode == "fwd":
                rv = ok & (x.index >= "2026-10-14")
                review[label] = (p[rv] - x.price[rv]).groupby(x.index[rv].tz_convert("GB").date).mean()
res = pd.DataFrame(rows).set_index(["eval", "set"]).round(2)
pd.set_option("display.width", 200)
print(res.sort_index().to_string())
print("\n== fwd daily bias, 14-30 Oct ==")
print(pd.DataFrame(review).round(0).to_string())
res.to_csv(f"{D}/structural_results.csv")
