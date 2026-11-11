"""E6: does the published low/high band cover the extremes? Region G (p/kWh), 2-4 d ahead, target dates from 14 Oct.
The band is p10/p90 of holdout residuals by horizon, widened by the weather ensemble (update.py), so nominal cover is 80%+.
Run: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/band.py
"""
import pandas as pd
from config.utils import day_ahead_to_agile
from prices.models import AgileData, PriceHistory

REGION = "G"
q = AgileData.objects.filter(region=REGION, date_time__gte="2026-10-14").values("forecast__created_at", "date_time", "agile_pred", "agile_low", "agile_high")
b = pd.DataFrame(q).rename(columns={"forecast__created_at": "created_at"})
a = pd.DataFrame(PriceHistory.objects.filter(date_time__gte="2026-10-14").values("date_time", "day_ahead")).set_index("date_time").day_ahead
a = pd.Series(day_ahead_to_agile(a.copy(), region=REGION).values, index=a.index)  # outturn in region G terms
b = b.join(a.rename("actual"), on="date_time").dropna(subset=["actual"])
b["h"] = (b.date_time - b.created_at).dt.total_seconds() / 86400
loc = b.date_time.dt.tz_convert("Europe/London")
b["gbdate"], b["hour"] = loc.dt.tz_localize(None).dt.normalize(), loc.dt.hour
s = pd.read_pickle("analysis/forecast_review_2026_11/data/daily.pkl").surplus_L2 / 1000
b = b.join(s.rename("surplus"), on="gbdate")
b["below"], b["above"] = b.actual < b.agile_low, b.actual > b.agile_high
b["width"] = b.agile_high - b.agile_low

def row(g):
    return pd.Series(dict(slots=len(g), days=g.gbdate.nunique(), inside=1 - g.below.mean() - g.above.mean(), below=g.below.mean(), above=g.above.mean(),
                          width=g.width.mean(), over_by=(g.actual - g.agile_high)[g.above].mean(), under_by=(g.agile_low - g.actual)[g.below].mean()))

pd.set_option("display.width", 250)
print("by horizon")
print(b.groupby(pd.cut(b.h, [1, 2, 4, 7, 15])).apply(row).round(2).to_string())
m = b[(b.h >= 2) & (b.h < 4)]
print("\n2-4 d, evening peak (16-19) by NESO 2-day surplus (GW)")
print(m[m.hour.between(16, 18)].groupby(pd.cut(m.surplus, [-10, 6, 8, 10, 30])).apply(row).round(2).to_string())
print("\n2-4 d, by actual Agile price (p/kWh)")
print(m.groupby(pd.cut(m.actual, [-10, 5, 15, 25, 35, 100])).apply(row).round(2).to_string())
