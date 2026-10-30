"""Dump stored forecasts and settled prices for the late-October 2026 review.

Run on the CT:  .venv/bin/python manage.py shell < analysis/forecast_review_2026_10/extract.py
Writes CSVs to analysis/forecast_review_2026_10/data/ (gitignored).
"""
import os

import pandas as pd

from prices.models import ForecastData, Forecasts, PriceHistory

OUT = "analysis/forecast_review_2026_10/data"
os.makedirs(OUT, exist_ok=True)

START = "2026-10-01"

prices = pd.DataFrame(PriceHistory.objects.filter(date_time__gte=START).values("date_time", "day_ahead", "agile"))
prices.to_csv(f"{OUT}/prices.csv", index=False)

fcs = pd.DataFrame(Forecasts.objects.filter(name__gte=START).values("id", "name"))
fd = pd.DataFrame(
    ForecastData.objects.filter(forecast_id__in=list(fcs["id"]))
    .values()
)
fd = fd.drop(columns=["shap_top_features"]).merge(fcs.rename(columns={"id": "forecast_id"}), on="forecast_id")
fd.to_csv(f"{OUT}/forecast_data.csv", index=False)
print(len(prices), "prices;", len(fcs), "forecasts;", len(fd), "forecast rows")
