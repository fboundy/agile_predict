"""Score stored forecasts against settled day-ahead prices, by horizon and by day."""
import numpy as np
import pandas as pd

D = "analysis/forecast_review_2026_10/data"
p = pd.read_csv(f"{D}/prices.csv", parse_dates=["date_time"]).set_index("date_time")
fd = pd.read_csv(f"{D}/forecast_data.csv", parse_dates=["date_time"])
fd["created_at"] = pd.to_datetime(fd["name"]).dt.tz_localize("GB", ambiguous="NaT").dt.tz_convert("UTC")
fd["dt"] = (fd["date_time"] - fd["created_at"]).dt.total_seconds() / 86400
fd = fd.merge(p[["day_ahead"]].rename(columns={"day_ahead": "actual"}), left_on="date_time", right_index=True)
fd["err"] = fd["day_ahead"] - fd["actual"]
fd["gb_date"] = fd["date_time"].dt.tz_convert("GB").dt.date

# Daily actual price profile
p["gb_date"] = p.index.tz_convert("GB").date
daily = p[p.index >= "2026-10-14"].groupby("gb_date")["day_ahead"].agg(["mean", "min", "max", "std"])
print("== Settled day-ahead (£/MWh) by GB day ==")
print(daily.round(1).to_string())

fd = fd[fd["date_time"] >= "2026-10-14"]
bins = [0, 1, 2, 3, 5, 8, 15]
fd["hbin"] = pd.cut(fd["dt"], bins)
def stats(g):
    return pd.Series({
        "n": len(g), "bias": g["err"].mean(), "mae": g["err"].abs().mean(),
        "rmse": np.sqrt((g["err"] ** 2).mean()),
        "sd_ratio": g["day_ahead"].std() / g["actual"].std(),
        "corr": g["day_ahead"].corr(g["actual"]),
    })
print("\n== Error by horizon (days) ==")
print(fd.groupby("hbin", observed=True).apply(stats).round(2).to_string())

h = fd[(fd["dt"] >= 2) & (fd["dt"] < 4)]
print("\n== 2-4 d horizon, by target GB day ==")
print(h.groupby("gb_date").apply(lambda g: pd.concat([stats(g), pd.Series({"pred_mean": g.day_ahead.mean(), "act_mean": g.actual.mean(), "pred_min": g.day_ahead.min(), "act_min": g.actual.min(), "pred_max": g.day_ahead.max(), "act_max": g.actual.max()})])).round(1).to_string())
fd.to_pickle(f"{D}/scored.pkl")
