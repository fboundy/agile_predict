"""Settled-price range in the true 60 days before each extreme day (PriceHistory, not the truncated extract)."""
import pandas as pd
from prices.models import PriceHistory
p = pd.DataFrame(PriceHistory.objects.filter(date_time__gte="2026-08-01").values("date_time", "day_ahead")).set_index("date_time").day_ahead.sort_index()
for d, kind in [("2026-10-24", "min"), ("2026-10-25", "min"), ("2026-10-29", "mean"), ("2026-11-06", "max"), ("2026-11-09", "max"), ("2026-11-10", "max")]:
    t = pd.Timestamp(d, tz="GB")
    w = p[(p.index >= t - pd.Timedelta(days=60)) & (p.index < t)]
    day = p[(p.index >= t) & (p.index < t + pd.Timedelta(days=1))]
    v = getattr(day, kind)()
    beyond = (w <= v).sum() if kind == "min" else (w >= v).sum()
    print(f"{d} day {kind}={v:.0f} | 60d window min={w.min():.0f} p1={w.quantile(.01):.0f} p99={w.quantile(.99):.0f} max={w.max():.0f} | window slots at/beyond day value: {beyond} | window days with mean>=day mean: {(w.groupby(w.index.tz_convert('GB').date).mean() >= day.mean()).sum()}")
sep = p["2026-09-01":"2026-09-30"]
top = sep[sep > 300]
print("\nSept slots > £300:", len(top), "on", sorted(set(top.index.tz_convert("GB").date)))
