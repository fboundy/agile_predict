"""Replay the shadow candidates over the forecasts already stored on dev (docs/FORECAST_REVIEW_2026_11.md §15).

For each day's training-slot forecast F: train on the forecasts made before F with the prices known
at F, then predict F's own stored inputs with shadow_predictions, the code the live shadow runs.
Offline, so the four columns compare with each other, not with production (MODEL_DYNAMIC_RANGE.md).
The dev DB starts 18 Sep, so the training window grows from about 25 days to 54, not a fixed 60.

Run on the dev box: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/shadow_replay.py
then: SHADOW_DIR=analysis/forecast_review_2026_11/data/replay .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/shadow_score.py
"""
import os

import pandas as pd
from prices.forecast_features import build_forecast_frame, resolve_feature_columns, select_daily_training_forecasts
from prices.management.commands.update import SHADOW_EXTRAS, shadow_predictions
from prices.models import ForecastData, Forecasts, PriceHistory

OUT = "analysis/forecast_review_2026_11/data/replay"
os.makedirs(OUT, exist_ok=True)
df, ff = build_forecast_frame(pd.DataFrame(list(ForecastData.objects.values())), pd.DataFrame(list(Forecasts.objects.values())))
prices = pd.DataFrame(PriceHistory.objects.filter(date_time__gte="2026-09-01").values("date_time", "day_ahead")).set_index("date_time")
feats = resolve_feature_columns("full_fr_eu")  # dev's live set
daily = select_daily_training_forecasts(ff)
for fid, t in daily.created_at[daily.created_at >= "2026-10-12"].items():
    out_file = f"{OUT}/{t:%Y-%m-%d %H%M}.csv"
    if os.path.exists(out_file):
        continue
    past = df[df.created_at < t].copy()
    past["days_ago"] = (t - past.created_at).dt.total_seconds() / 86400
    known = prices[prices.index < t.normalize() + pd.Timedelta(hours=47)]  # day-ahead for tomorrow is out by the afternoon run
    fc = df[df.forecast_id == fid].assign(days_ago=0)
    out = shadow_predictions(past, select_daily_training_forecasts(ff[ff.created_at < t]), known, feats, 60, fc, {"base": [], **SHADOW_EXTRAS})
    out["opmr_national_surplus"] = fc.opmr_national_surplus
    out.to_csv(out_file)
    print(t, len(out), flush=True)
