"""E1: is the price-vs-residual-demand curve stable across winters once gas is divided out?

Run: .venv/bin/python analysis/forecast_review_2026_11/stack.py
"""
import numpy as np
import pandas as pd

p = pd.read_pickle("analysis/forecast_review_2026_11/data/panel.pkl")
p["ratio"] = p.day_ahead / p.gas_ttf
gb = p.index.tz_convert("Europe/London")
p["hour"] = gb.hour + gb.minute / 60
m, y = p.index.month, p.index.year
p["season"] = np.where(m >= 10, y.astype(str), np.where(m <= 3, (y - 1).astype(str), "summer"))
p.loc[p.season != "summer", "season"] = "W" + p.season + "/" + (p.loc[p.season != "summer", "season"].astype(int) + 1).astype(str).str[-2:]
p["rbin"] = pd.cut(p.resid / 1000, [-5, 10, 15, 20, 25, 30, 33, 36, 50])
w = p[p.season != "summer"]

pd.set_option("display.width", 250)
print("slots per residual-demand bin (GW)")
print(w.pivot_table(index="season", columns="rbin", values="ratio", aggfunc="size", observed=False).fillna(0).astype(int).to_string())
print("\nmedian price / TTF")
print(w.pivot_table(index="season", columns="rbin", values="ratio", aggfunc="median", observed=False).round(2).to_string())
print("\n90th percentile price / TTF")
print(w.pivot_table(index="season", columns="rbin", values="ratio", aggfunc=lambda s: s.quantile(0.9), observed=False).round(2).to_string())
print("\nmedian price, GBP/MWh")
print(w.pivot_table(index="season", columns="rbin", values="day_ahead", aggfunc="median", observed=False).round(0).to_string())
print("\nmean TTF by season:", w.groupby("season").gas_ttf.mean().round(1).to_dict())

# price = a + b * TTF within a residual-demand band: is there a fixed (carbon) offset?
print("\nprice ~ a + b*TTF by residual-demand band, all data")
for lo, hi in [(10, 15), (15, 20), (20, 25), (25, 30), (30, 36)]:
    s = p[(p.resid >= lo * 1000) & (p.resid < hi * 1000)]
    b, a = np.polyfit(s.gas_ttf, s.day_ahead, 1)
    print(f"  {lo}-{hi} GW: n={len(s):6d}  a={a:6.1f}  b={b:5.2f}  -> offset c=a/b={a / b:5.1f} EUR/MWh  r={s.gas_ttf.corr(s.day_ahead):.2f}")

print("\n9 Nov 2026 evening")
print(p.loc["2026-11-09 15:30":"2026-11-09 19:30", ["day_ahead", "resid", "wind", "gas_ttf", "ratio", "imports", "gas"]].round(2).to_string())
print("\nevery earlier slot with residual demand >= 36 GW: price/TTF distribution by season")
t = p[(p.resid >= 36000) & (p.index < "2026-10-01")]
print(t.groupby("season").ratio.describe().round(2).to_string())
print("\ntop 15 price/TTF slots in the panel")
print(p.nlargest(15, "ratio")[["day_ahead", "resid", "gas_ttf", "ratio", "hour"]].round(2).to_string())
