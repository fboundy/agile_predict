"""Score dev-box forecasts on the extremes, 14 Oct - 11 Nov 2026 (GB days).

Run on the dev box: .venv/bin/python analysis/forecast_review_2026_11/score.py
Model era: before 2026-10-30 17:33 UTC = default set (prod-equivalent); after = full_fr_eu candidate.
"""
import numpy as np
import pandas as pd

D = "analysis/forecast_review_2026_11/data"
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
p = pd.read_csv(f"{D}/prices.csv", parse_dates=["date_time"]).set_index("date_time").sort_index()
fc = pd.read_csv(f"{D}/forecasts.csv", parse_dates=["created_at"])
fd = pd.read_csv(f"{D}/forecast_data.csv", parse_dates=["date_time"])
fd = fd.merge(fc.rename(columns={"id": "forecast_id"})[["forecast_id", "created_at"]], on="forecast_id")
fd["dt"] = (fd.date_time - fd.created_at).dt.total_seconds() / 86400
fd = fd[fd.dt > 0].merge(p[["day_ahead"]].rename(columns={"day_ahead": "actual"}), left_on="date_time", right_index=True)
fd["err"] = fd.day_ahead - fd.actual
fd["gb_date"] = fd.date_time.dt.tz_convert("GB").dt.date
fd["era"] = np.where(fd.created_at < pd.Timestamp("2026-10-30 17:33Z"), "default", "eu_cand")
fd = fd[(fd.date_time >= "2026-10-14") & (fd.date_time < "2026-11-11T23:00Z")]
fd.to_pickle(f"{D}/scored.pkl")

a = p.loc["2026-10-14":"2026-11-11", "day_ahead"]
a_day = a.groupby(a.index.tz_convert("GB").date)
print("== Settled day-ahead £/MWh by GB day ==")
print(pd.DataFrame({"mean": a_day.mean(), "min": a_day.min(), "max": a_day.max(), "neg_slots": a_day.apply(lambda s: (s < 0).sum()),
                    "spread": a_day.max() - a_day.min()}).round(0).to_string())
lo, hi = a.quantile(0.10), a.quantile(0.90)
print(f"\nwindow: n={len(a)} p10={lo:.0f} p90={hi:.0f} min={a.min():.0f} max={a.max():.0f} sd={a.std():.0f}")

fd["tail"] = np.select([fd.actual <= lo, fd.actual >= hi], ["low10", "high10"], "mid")
fd["hbin"] = pd.cut(fd.dt, [0, 1, 2, 3, 5, 8, 15])
def stats(g):
    return pd.Series({"n": len(g), "bias": g.err.mean(), "mae": g.err.abs().mean(),
                      "sd_ratio": g.day_ahead.std() / g.actual.std() if len(g) > 2 else np.nan})
print("\n== By horizon x tail (all eras) ==")
print(fd.groupby(["hbin", "tail"], observed=True).apply(stats).round(2).unstack("tail").to_string())
print("\n== By era x tail, 2-5 d horizon ==")
h = fd[(fd.dt >= 2) & (fd.dt < 5)]
print(h.groupby(["era", "tail"]).apply(stats).round(2).to_string())

# Daily extremes: forecast daily max/min vs actual, per (forecast run, GB day), averaged over 2-4 d runs
g = fd[(fd.dt >= 2) & (fd.dt < 4)].groupby(["forecast_id", "gb_date"])
dd = pd.DataFrame({"pmax": g.day_ahead.max(), "amax": g.actual.max(), "pmin": g.day_ahead.min(), "amin": g.actual.min(),
                   "pmean": g.day_ahead.mean(), "amean": g.actual.mean(), "n": g.size(), "era": g.era.first()})
dd = dd[dd.n >= 40].groupby("gb_date").agg({"amean": "first", "pmean": "mean", "amax": "first", "pmax": "mean", "amin": "first", "pmin": "mean", "era": "last"})
dd["max_err"] = dd.pmax - dd.amax
dd["min_err"] = dd.pmin - dd.amin
dd["mean_err"] = dd.pmean - dd.amean
print("\n== Daily extremes, 2-4 d ahead (avg over runs) ==")
print(dd.round(0).to_string())

# Peak (16-19 GB) and overnight trough
fd["hour"] = fd.date_time.dt.tz_convert("GB").dt.hour
print("\n== 2-5 d: peak 16-19h and 00-06h, by era ==")
for nm, m in [("peak16-19", fd.hour.between(16, 18)), ("night00-06", fd.hour.between(0, 5))]:
    print(nm, fd[m & (fd.dt >= 2) & (fd.dt < 5)].groupby("era").apply(stats).round(2).to_string().replace("\n", " | "))
