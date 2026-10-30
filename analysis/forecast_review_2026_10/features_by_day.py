"""Daily mean feature values: as forecast 2-4 d ahead vs the freshest (<1 d) forecast."""
import pandas as pd

D = "analysis/forecast_review_2026_10/data"
fd = pd.read_pickle(f"{D}/scored.pkl")
fd["total_wind"] = fd["bm_wind"] + fd["emb_wind"]
fd["resid"] = fd["demand"] - fd["bm_wind"] - fd["solar"] - fd["nuclear"]  # demand is transmission-level (net of embedded)
cols = ["bm_wind", "emb_wind", "solar", "demand", "nuclear", "dispatchable_capacity", "fr_wind", "gas_ttf", "resid", "day_ahead", "actual"]
far = fd[(fd.dt >= 2) & (fd.dt < 4)].groupby("gb_date")[cols].mean()
near = fd[(fd.dt > 0) & (fd.dt < 1)].groupby("gb_date")[cols].mean()
pd.set_option("display.width", 250)
print("== 2-4 d ahead (GW where MW) ==")
print((far.assign(**{c: far[c] / 1000 for c in cols[:7] + ["resid"]})).round(2).to_string())
print("\n== <1 d ahead ==")
print((near.assign(**{c: near[c] / 1000 for c in cols[:7] + ["resid"]})).round(2).to_string())
print("\n== bm_wind / emb_wind ratio (near) ==")
print((near.bm_wind / near.emb_wind).round(2).to_string())
