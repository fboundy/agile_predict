"""E4: walk-forward test of model forms on the three-year panel.

Inputs are ACTUAL drivers (perfect weather/demand foresight) plus NESO's OPMR at a fixed lead, and the
model is LightGBM alone, not the production ensemble. So the absolute numbers are not production's;
only the differences between variants mean anything, and a win here is a reason to test live, no more.

Weekly refit: train on rows before T, predict [T+2d, T+9d) (the latest known price is 2+ days old,
like a 2+ day forecast). Variants = window x target x feature group.
Run: .venv/bin/python analysis/forecast_review_2026_11/modelform.py   (fits once, then re-scores from data/e4_preds.pkl)
"""
import os

import numpy as np
import pandas as pd

D = "analysis/forecast_review_2026_11/data"
C = 12.0  # fixed offset in price ~ a + b*TTF (stack.py)
BASE = ["hour", "weekend", "solar", "wind_emb", "demand", "peak", "nuclear", "gas_ttf", "eu_wind_cf", "disp_L2", "days_ago"]
GROUPS = {
    "base": BASE,
    "noeu": [f for f in BASE if f != "eu_wind_cf"],
    "sur": BASE + ["surplus_L2"],
    "sur4": BASE + ["surplus_L4"],
    "margin": BASE + ["surplus_L2", "resid", "headroom"],
    "resid": BASE + ["resid"],      # residual demand as one explicit feature
    "bmw": BASE + ["bm_wind"],      # the same information as a raw feature (the form that lost its live trial)
}
VARIANTS = [(w, t, g) for w in (60, 365, 0) for t in ("raw", "norm") for g in ("base", "sur", "margin")]
VARIANTS += [(60, "raw", "noeu"), (60, "raw", "sur4"), (0, "norm", "sur4")]
VARIANTS += [(w, t, g) for w, t in ((60, "raw"), (0, "norm")) for g in ("resid", "bmw")]
name = lambda v: f"{v[0] or 'all'}/{v[1]}/{v[2]}"

p = pd.read_pickle(f"{D}/panel.pkl")
loc = p.index.tz_convert("Europe/London")
p["gbdate"] = loc.tz_localize(None).normalize()
p = p.join(pd.read_pickle(f"{D}/opmr.pkl")[["surplus_L2", "surplus_L4", "disp_L2"]], on="gbdate")
p["hour"] = loc.hour + loc.minute / 60
p["weekend"] = (loc.dayofweek >= 5).astype(int)
p["demand"] = p.generation - p.wind_emb - p.solar
p["peak"] = p.groupby("gbdate").demand.transform("max")
p["headroom"] = p.disp_L2 - p.demand
p["bm_wind"] = p.wind - p.wind_emb
p = p.dropna(subset=["day_ahead", "gas_ttf"])
p["norm"] = p.day_ahead / (p.gas_ttf + C)

done = pd.read_pickle(f"{D}/e4_preds.pkl") if os.path.exists(f"{D}/e4_preds.pkl") else None
todo = [v for v in VARIANTS if done is None or name(v) not in done.columns]
if todo:
    import lightgbm as lgb

    out = []
    for T in pd.date_range(pd.Timestamp("2024-10-01", tz=p.index.tz), p.index.max() - pd.Timedelta(days=2), freq="7D"):
        te = p[(p.index >= T + pd.Timedelta(days=2)) & (p.index < T + pd.Timedelta(days=9))].assign(days_ago=0.0)
        tr_all = p[p.index < T].copy()
        tr_all["days_ago"] = (T - tr_all.index).total_seconds() / 86400
        res = te[["day_ahead"]].copy()
        for v in todo:
            win, tgt, grp = v
            tr = tr_all[tr_all.days_ago <= win] if win else tr_all
            y = tr.norm if tgt == "norm" else tr.day_ahead
            w = np.maximum(1, ((y - y.mean()) / y.std()).abs())  # production's weighting
            if win != 60:
                w = w * 0.5 ** (tr.days_ago / 365)  # recency: a year-old row counts half
            m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, n_jobs=2, random_state=0, verbose=-1)
            m.fit(tr[GROUPS[grp]], y, sample_weight=w)
            pr = m.predict(te[GROUPS[grp]])
            res[name(v)] = pr * (te.gas_ttf + C) if tgt == "norm" else pr
        out.append(res)
        print(T.date(), len(te), flush=True)
    out = pd.concat(out)
    (out if done is None else done.join(out.drop(columns="day_ahead"))).to_pickle(f"{D}/e4_preds.pkl")

r = pd.read_pickle(f"{D}/e4_preds.pkl").join(p[["gbdate", "gas_ttf", "surplus_L2"]])
cols = [name(v) for v in VARIANTS]
assert r[cols].notna().all().all() and r.index.is_unique
# Averaging a short and a long calibration window costs no refit, so score it too.
for tg in ("raw/base", "norm/base", "norm/sur", "norm/margin"):
    r[f"60+all/{tg}"] = (r[f"60/{tg}"] + r[f"all/{tg}"]) / 2
    cols.append(f"60+all/{tg}")
a = r.day_ahead
ref = "60/raw/base"
wk = r.index.to_period("W")
g = r.groupby("gbdate")
day = g[["day_ahead"] + cols].max().join(g.gas_ttf.mean()).join(g.surplus_L2.first())
day = day[g.size() >= 46]
nact = day.day_ahead / (day.gas_ttf + C)
thr = nact.quantile(0.95)
spike = nact >= thr
hi, lo = a >= a.quantile(0.95), a <= a.quantile(0.05)

from sklearn.metrics import roc_auc_score

rows = {}
for c in cols:
    e = r[c] - a
    nfc = day[c] / (day.gas_ttf + C)
    flag = nfc >= nfc.quantile(0.95)
    wse = (e ** 2).groupby(wk).mean()
    rows[c] = dict(
        MAE=e.abs().mean(), RMSE=np.sqrt((e ** 2).mean()), slope=np.polyfit(r[c], a, 1)[0],
        top5_err=e[hi].mean(), bot5_err=e[lo].mean(),
        low_recall=(r[c][a <= 10] <= 10).mean(), low_prec=(a[r[c] <= 10] <= 10).mean(),
        pk_MAE=(day[c] - day.day_ahead).abs().mean(), spike_AUC=roc_auc_score(spike, nfc), spike_hit=(flag & spike).sum() / spike.sum(),
        spike_pk_err=(day[c] - day.day_ahead)[spike].mean(),
        spike_recall=(nfc[spike] >= thr).mean(), false_spikes=(nfc[~spike] >= thr).sum(),  # at the actual threshold, not rank
        wks_won=(wse < ((r[ref] - a) ** 2).groupby(wk).mean()).mean(),
    )
pd.set_option("display.width", 250)
t = pd.DataFrame(rows).T
print(f"test slots {len(r)}, {r.index.min():%d %b %Y} -> {r.index.max():%d %b %Y}; days {len(day)}, spike days {spike.sum()}; slots <= 10: {(a <= 10).sum()}")
print(t.round(2).to_string())

m, yr = r.index.month, r.index.year
r["season"] = np.where(m >= 10, "W" + yr.astype(str), np.where(m <= 3, "W" + (yr - 1).astype(str), "S" + yr.astype(str)))
print("\nRMSE by season")
print(pd.DataFrame({c: np.sqrt(((r[c] - a) ** 2).groupby(r.season).mean()) for c in cols}).T.round(1).to_string())

print("\nforecast daily peak on named days (actual first)")
print(day.loc[day.index.isin(pd.to_datetime(["2025-01-08", "2026-06-24", "2026-11-06", "2026-11-09"])), ["day_ahead"] + cols].T.round(0).to_string())

tight = day.surplus_L2 < 4000
print(f"\ndaily-peak error on days with 2-d surplus < 4 GW (n={tight.sum()}) vs the rest: mean error, MAE")
print(pd.DataFrame({c: [(day[c] - day.day_ahead)[tight].mean(), (day[c] - day.day_ahead)[tight].abs().mean(),
                        (day[c] - day.day_ahead)[~tight].mean(), (day[c] - day.day_ahead)[~tight].abs().mean()] for c in cols},
                   index=["tight err", "tight MAE", "rest err", "rest MAE"]).T.round(1).to_string())
