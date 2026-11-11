"""E3: anatomy of the stored forecasts' errors at 2-4 days (no retraining, so no offline-replay caveat).

Run: .venv/bin/python analysis/forecast_review_2026_11/forensics.py
"""
import numpy as np
import pandas as pd

D = "analysis/forecast_review_2026_11/data"
fd = pd.read_csv(f"{D}/forecast_data.csv", parse_dates=["date_time"])
ff = pd.read_csv(f"{D}/forecasts.csv", index_col="id")
ff["created_at"] = pd.to_datetime(ff.created_at, utc=True, format="mixed")
pr = pd.read_csv(f"{D}/prices.csv", parse_dates=["date_time"]).set_index("date_time")["day_ahead"].rename("actual")
df = fd.merge(ff[["created_at"]], left_on="forecast_id", right_index=True).join(pr, on="date_time", how="inner")
df["h"] = (df.date_time - df.created_at).dt.total_seconds() / 86400
df = df[df.date_time >= "2026-10-14"]
df["err"] = df.day_ahead - df.actual
gb = df.date_time.dt.tz_convert("Europe/London")
df["gbdate"] = gb.dt.tz_localize(None).dt.normalize()
df["hour"] = gb.dt.hour
df["resid_fc"] = df.demand - df.bm_wind - df.emb_wind - df.solar - df.nuclear
df["margin_fc"] = df.dispatchable_capacity - df.demand
pd.set_option("display.width", 250)

print("== (a) calibration: actual = a + b * forecast. b > 1 means the forecast is too flat ==")
for lo, hi in [(1, 2), (2, 4), (4, 7), (7, 15)]:
    s = df[(df.h >= lo) & (df.h < hi)]
    b, a = np.polyfit(s.day_ahead, s.actual, 1)
    print(f"  {lo}-{hi} d: n={len(s):6d}  slope {b:.2f}  intercept {a:+6.1f}  sd(fc)/sd(actual) {s.day_ahead.std() / s.actual.std():.2f}  corr {s.day_ahead.corr(s.actual):.2f}"
          f"  RMSE {np.sqrt((s.err ** 2).mean()):.1f}  RMSE after this in-sample stretch {np.sqrt(((a + b * s.day_ahead - s.actual) ** 2).mean()):.1f}")

s = df[(df.h >= 2) & (df.h < 4)].copy()
print("\n== (b) 2-4 d error split into the day's level and the within-day shape ==")
g = s.groupby(["forecast_id", "gbdate"])
s["lvl"] = g.err.transform("mean")
s["shp"] = s.err - s.lvl
print(f"  MSE {np.mean(s.err ** 2):.0f} = level {np.mean(s.lvl ** 2):.0f} ({np.mean(s.lvl ** 2) / np.mean(s.err ** 2):.0%}) + shape {np.mean(s.shp ** 2):.0f}")
s["fc_range"] = g.day_ahead.transform("max") - g.day_ahead.transform("min")
s["ac_range"] = g.actual.transform("max") - g.actual.transform("min")
r = s.drop_duplicates(["forecast_id", "gbdate"])
print(f"  within-day range (max - min): forecast median {r.fc_range.median():.0f}, actual median {r.ac_range.median():.0f}; corr {r.fc_range.corr(r.ac_range):.2f}")
print("  mean error by GB hour:", s.groupby("hour").err.mean().round(0).astype(int).to_dict())

print("\n== (c) persistence of the daily level error (one value per target day: mean over 2-4 d runs) ==")
de = s.groupby("gbdate").lvl.mean()
print("  autocorrelation at lag 1..6 days:", [round(de.autocorr(k), 2) for k in range(1, 7)])

print("\n== (d) what the 2-4 d error correlates with (stored inputs the model did not use are marked *) ==")
cols = {"opmr_national_surplus": "*NESO surplus", "margin_fc": "*dispatchable - demand", "gas_availability": "*gas availability", "bm_wind": "*bm_wind",
        "resid_fc": "*residual demand (derived)", "temp_2m": "*temp", "eu_wind_cf": "eu_wind_cf (* before 30 Oct)", "gas_ttf": "gas_ttf", "demand": "demand",
        "emb_wind": "emb_wind", "dispatchable_capacity": "dispatchable_capacity"}
pk = s[(s.hour >= 16) & (s.hour < 19)]
dd = s.groupby("gbdate").agg(err=("err", "mean"), **{c: (c, "mean") for c in cols})
dpk = pk.groupby("gbdate").agg(err=("err", "mean"), **{c: (c, "mean") for c in cols})
out = pd.DataFrame({"all slots": [s.err.corr(s[c]) for c in cols], "daily mean": [dd.err.corr(dd[c]) for c in cols],
                    "evening peak, daily": [dpk.err.corr(dpk[c]) for c in cols]}, index=cols.values())
print(out.round(2).to_string())

print("\n== (e) evening-peak (16-19) error by NESO surplus band, 2-4 d ==")
pk = pk.assign(sb=pd.cut(pk.opmr_national_surplus / 1000, [-10, 4, 6, 8, 10, 30]))
print(pk.groupby("sb", observed=False).agg(slots=("err", "size"), days=("gbdate", "nunique"), mean_err=("err", "mean"), actual=("actual", "mean"), forecast=("day_ahead", "mean")).round(1).to_string())
s.to_pickle(f"{D}/h24.pkl")

print("\n== (f) carry the last known day's level error forward: daily-level RMSE before -> after (in-sample coefficient, so an upper bound) ==")
for lag in (1, 2, 3):
    x = pd.concat([de, de.shift(lag)], axis=1, keys=["e", "prev"]).dropna()
    k = (x.e * x.prev).sum() / (x.prev ** 2).sum()
    print(f"  lag {lag} d: k={k:.2f}  RMSE {np.sqrt((x.e ** 2).mean()):.1f} -> {np.sqrt(((x.e - k * x.prev) ** 2).mean()):.1f}")

print("\n== (g) daily error correlation with eu_wind_cf by era (feature unused before 30 Oct 17:33, used after) ==")
s["era"] = np.where(s.created_at < pd.Timestamp("2026-10-30 17:33", tz="UTC"), "default set", "full_fr_eu")
for era, g in s.groupby("era"):
    d2 = g.groupby("gbdate").agg(err=("err", "mean"), eu=("eu_wind_cf", "mean"), emb=("emb_wind", "mean"), sur=("opmr_national_surplus", "mean"))
    print(f"  {era}: days={len(d2)}  eu_wind_cf {d2.err.corr(d2.eu):+.2f}  emb_wind {d2.err.corr(d2.emb):+.2f}  NESO surplus {d2.err.corr(d2.sur):+.2f}  mean err {d2.err.mean():+.1f}")
