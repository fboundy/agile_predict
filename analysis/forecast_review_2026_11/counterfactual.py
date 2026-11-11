"""Rolling offline test: could a model that extrapolates in residual demand have caught the Nov spikes?

Same as-forecast inputs for every variant; train on 22-46 h rows from the prior 60 d (data only kept from
12 Sep, so early folds are shorter), predict each day's 04:15 run at 2-4 d lead. Compare variants with
each other, NOT with production (offline replays have not reproduced production: MODEL_DYNAMIC_RANGE.md).
Run on the dev box: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/counterfactual.py
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from prices.models import ForecastData, PriceHistory, Forecasts

pd.set_option("display.width", 250)
fc = pd.DataFrame(Forecasts.objects.filter(created_at__gte="2026-09-01").values("id", "created_at"))
cols = ["forecast_id", "date_time", "demand", "bm_wind", "emb_wind", "solar", "gas_ttf", "eu_wind_cf", "dispatchable_capacity"]
f = pd.DataFrame(ForecastData.objects.filter(forecast_id__in=list(fc.id)).values(*cols)).merge(fc.rename(columns={"id": "forecast_id"}), on="forecast_id")
p = pd.DataFrame(PriceHistory.objects.filter(date_time__gte="2026-09-01").values("date_time", "day_ahead")).set_index("date_time").day_ahead
f["lead_h"] = (f.date_time - f.created_at).dt.total_seconds() / 3600
f["price"] = f.date_time.map(p)
gb = f.date_time.dt.tz_convert("GB")
f["tod"] = gb.dt.hour + gb.dt.minute / 60
f["weekend"] = (gb.dt.dayofweek >= 5).astype(int)
f["resid"] = (f.demand - f.bm_wind - f.emb_wind - f.solar) / 1000
f["peak"] = f.tod.between(16, 19.5).astype(int)
f["resid_pk"] = f.resid * f.peak
f["gb_date"] = gb.dt.date
X = ["resid", "demand", "bm_wind", "emb_wind", "solar", "gas_ttf", "eu_wind_cf", "dispatchable_capacity", "tod", "weekend", "peak"]
f[X] = f[X].astype(float)
train_pool = f[f.lead_h.between(22, 46)].sort_values("created_at").drop_duplicates("date_time", keep="last").dropna(subset=["price"])
med = train_pool[X].median()

def variants():
    yield "ET (price)", lambda: ExtraTreesRegressor(300, min_samples_leaf=3, n_jobs=2, random_state=0), False
    yield "ET (price/gas)", lambda: ExtraTreesRegressor(300, min_samples_leaf=3, n_jobs=2, random_state=0), True
    yield "LGBM linear_tree (price)", lambda: LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, linear_tree=True, verbose=-1, random_state=0), False
    yield "Ridge resid+tod (price/gas)", "ridge", True

runs = fc[(fc.created_at >= "2026-10-10") & (fc.created_at.dt.tz_convert("GB").dt.hour == 4)].sort_values("created_at")
out = []
for _, run in runs.iterrows():
    tr = train_pool[(train_pool.date_time < run.created_at) & (train_pool.date_time >= run.created_at - pd.Timedelta(days=60))]
    te = f[(f.forecast_id == run.id) & (f.lead_h.between(48, 96))].dropna(subset=["price"])
    if te.empty or len(tr) < 500:
        continue
    for name, mk, norm in variants():
        y = tr.price / tr.gas_ttf if norm else tr.price
        if mk == "ridge":
            # piecewise-linear in residual demand (hinges) per broad time-of-day, so price can rise past the training max
            def design(d):
                r = d.resid
                return pd.DataFrame({"r": r, "h30": (r - 30).clip(lower=0), "h34": (r - 34).clip(lower=0), "pk": d.peak, "rpk": r * d.peak,
                                     "we": d.weekend, "eu": d.eu_wind_cf.fillna(med.eu_wind_cf),
                                     **{f"t{h}": (d.tod // 3 == h).astype(int) for h in range(8)}})
            m = Ridge(alpha=1.0).fit(design(tr), y)
            pred = m.predict(design(te))
        else:
            m = mk().fit(tr[X].fillna(med), y)
            pred = m.predict(te[X].fillna(med))
        if norm:
            pred = pred * te.gas_ttf.values
        out.append(pd.DataFrame({"model": name, "date_time": te.date_time.values, "gb_date": te.gb_date.values, "pred": pred, "actual": te.price.values}))
r = pd.concat(out)
r = r[r.date_time >= pd.Timestamp("2026-10-14")]  # .values drops tz; values are UTC
r["err"] = r.pred - r.actual
lo, hi = r.actual.quantile([0.1, 0.9])
r["tail"] = np.select([r.actual <= lo, r.actual >= hi], ["low10", "high10"], "mid")
print(f"offline 2-4 d, target days 14 Oct-11 Nov; p10={lo:.0f} p90={hi:.0f}")
s = r.groupby(["model", "tail"]).err.agg(bias="mean", mae=lambda e: e.abs().mean()).unstack("tail")
s["mae_all"] = r.groupby("model").err.apply(lambda e: e.abs().mean())
s["rmse_all"] = r.groupby("model").err.apply(lambda e: np.sqrt((e ** 2).mean()))
print(s.round(1).to_string())
dm = r.groupby(["gb_date", "model"]).agg(pmax=("pred", "max"), amax=("actual", "max"), pmean=("pred", "mean"), amean=("actual", "mean"))
# pred per run differs; average across runs covering the day: approximate by mean of per-row predictions grouped above
t = dm.pmax.unstack("model").round(0)
t["actual_max"] = dm.amax.groupby("gb_date").first().round(0)
print("\n== Daily max (offline 2-4 d) ==")
print(t.loc[pd.Timestamp("2026-10-22").date():].to_string())
t2 = dm.pmean.unstack("model").round(0)
t2["actual_mean"] = dm.amean.groupby("gb_date").first().round(0)
print("\n== Daily mean (offline 2-4 d) ==")
print(t2.loc[pd.Timestamp("2026-10-22").date():].to_string())
r.to_pickle("analysis/forecast_review_2026_11/data/counterfactual.pkl")
