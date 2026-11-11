"""Was 9 Nov's evening tightness outside the 60-day training window, and does last winter contain it?

Run on the dev box: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/coverage.py
"""
import numpy as np
import pandas as pd
from prices.models import ForecastData, PriceHistory, Forecasts

pd.set_option("display.width", 250)
# Training-style rows: as forecast ~1-2 d ahead (22-46 h, as build_training_data keeps)
fc = pd.DataFrame(Forecasts.objects.filter(created_at__gte="2025-09-01").values("id", "created_at"))
print("forecast runs since 2025-09:", len(fc), "first:", fc.created_at.min())
rows = []
for chunk in np.array_split(fc.id.values, 40):
    rows += list(ForecastData.objects.filter(forecast_id__in=list(chunk)).values("forecast_id", "date_time", "demand", "bm_wind", "emb_wind", "solar", "gas_ttf"))
f = pd.DataFrame(rows).merge(fc.rename(columns={"id": "forecast_id"}), on="forecast_id")
f["lead_h"] = (f.date_time - f.created_at).dt.total_seconds() / 3600
f = f[f.lead_h.between(22, 46)].sort_values("created_at").drop_duplicates("date_time", keep="last").set_index("date_time").sort_index()
p = pd.DataFrame(PriceHistory.objects.filter(date_time__gte="2025-09-01").values("date_time", "day_ahead")).set_index("date_time")
f = f.join(p, how="inner")
f["resid"] = (f.demand - f.bm_wind - f.emb_wind) / 1000
f["month"] = f.index.tz_convert("GB").strftime("%Y-%m")
print("\n== Monthly: residual demand (GW, 22-46h forecast) and price ==")
print(f.groupby("month").agg(n=("resid", "size"), resid_p99=("resid", lambda s: s.quantile(.99)), resid_max=("resid", "max"),
                             price_mean=("day_ahead", "mean"), price_p99=("day_ahead", lambda s: s.quantile(.99)), price_max=("day_ahead", "max"),
                             ttf=("gas_ttf", "mean")).round(1).to_string())

def window(end, days=60):
    e = pd.Timestamp(end, tz="GB")
    return f[(f.index >= e - pd.Timedelta(days=days)) & (f.index < e)]
ev = f.loc[pd.Timestamp("2026-11-09 16:00", tz="GB"):pd.Timestamp("2026-11-09 19:30", tz="GB")]
print("\n9 Nov 16:00-19:30: resid GW", ev.resid.round(1).tolist(), "price", ev.day_ahead.round(0).tolist())
w = window("2026-11-09")
print("60 d window before 9 Nov: resid max", round(w.resid.max(), 1), " slots with resid >= 35 GW:", (w.resid >= 35).sum(),
      " their mean price:", round(w.day_ahead[w.resid >= 35].mean(), 0))
lw = f[(f.index >= "2025-11-01") & (f.index < "2026-03-01")]
for lo_, hi_ in [(33, 35), (35, 37), (37, 39), (39, 45)]:
    s = lw[(lw.resid >= lo_) & (lw.resid < hi_)]
    print(f"last winter (Nov-Feb) resid {lo_}-{hi_} GW: n={len(s)}, price mean {s.day_ahead.mean():.0f}, p90 {s.day_ahead.quantile(.9):.0f}, max {s.day_ahead.max():.0f}, ttf {s.gas_ttf.mean():.0f}")
