"""Score the shadow candidates against the run's own model (docs/FORECAST_REVIEW_2026_11.md §15).

The rule below was fixed on 11 Nov 2026, before the first live shadow run. Do not add a metric or
move a threshold here after seeing results: a candidate that fails this fails.

Run on the dev box: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/shadow_score.py
(SHADOW_DIR=<dir> scores another set of run files, e.g. the replay.)
"""
import glob
import os

import numpy as np
import pandas as pd
from prices.models import PriceHistory

D = os.environ.get("SHADOW_DIR", "logs/shadow")
CANDS = ["resid", "surplus", "resid_surplus"]
TIGHT_MW, LOW, ALARM = 6000, 20, 300      # tight day: stored NESO surplus under 6 GW; low slot: settles <= 20; alarm: forecast >= 300
NEED_TIGHT, NEED_LOW = 10, 5              # event budget (days)

runs = []
for f in sorted(glob.glob(f"{D}/*.csv")):
    r = pd.read_csv(f, index_col=0, parse_dates=True)
    made = pd.to_datetime(os.path.basename(f)[:-4], format="%Y-%m-%d %H%M").tz_localize("GB")
    lead = (r.index - made).total_seconds() / 86400
    runs.append(r[(lead >= 2) & (lead < 4)].assign(run=made))  # 0-1 d is GB60 passthrough, not the model
r = pd.concat(runs)
p = pd.DataFrame(PriceHistory.objects.filter(date_time__gte=r.index.min()).values("date_time", "day_ahead")).set_index("date_time").day_ahead
gbday = lambda i: i.tz_convert("GB").tz_localize(None).normalize()
r = r.join(p.rename("actual"), how="inner")
r["day"] = gbday(r.index)
hour = r.index.tz_convert("GB").hour
cols = ["base"] + CANDS
e = r[cols].sub(r.actual, axis=0)

tight = (hour >= 16) & (hour < 19) & r.day.isin(r.day[r.opmr_national_surplus < TIGHT_MW])
low = r.actual <= LOW
ordinary = r.day.map(p.groupby(gbday(p.index)).max()) < ALARM
by_day = lambda m: e[m].abs().groupby(r.day[m]).mean()
day_mse, hi, lo = (e ** 2).groupby(r.day).mean(), by_day(tight), by_day(low)
won = lambda d: (d[CANDS].lt(d.base, axis=0)).mean().reindex(cols)
t = pd.DataFrame({
    "RMSE": np.sqrt((e ** 2).mean()),
    "days_won": won(day_mse),
    "tight_eve_MAE": e[tight].abs().mean(), "tight_days_won": won(hi),
    "low_slot_mean_fc": r.loc[low, cols].mean(), "low_slot_MAE": e[low].abs().mean(), "low_days_won": won(lo),
    "false_alarms": r[ordinary].groupby(["run", "day"])[cols].max().ge(ALARM).sum(),
})
pd.set_option("display.width", 250)
print(f"{D}: {r.run.nunique()} runs, target days {r.day.min():%d %b} - {r.day.max():%d %b}, {len(r)} slot forecasts at 2-4 d")
print(f"event budget: tight days {len(hi)}/{NEED_TIGHT}, low days {len(lo)}/{NEED_LOW}")
print(t.round(2).to_string())
ok = (t.RMSE <= t.RMSE.base) & (t.tight_days_won > 0.5) & (t.low_days_won > 0.5) & (t.false_alarms <= t.false_alarms.base)
budget = len(hi) >= NEED_TIGHT and len(lo) >= NEED_LOW
for c in CANDS:
    print(f"{c}: {'PASS' if ok[c] else 'FAIL'}" + ("" if budget else " so far (event budget not met; not a verdict)"))
