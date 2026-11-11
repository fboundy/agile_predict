"""Dump dev-box forecasts, settled prices and external forecasts, 14 Oct - 11 Nov 2026.

Run on the dev box: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/extract.py
"""
import os
import pandas as pd
from prices.models import Forecasts, ForecastData, PriceHistory, ExternalForecast

OUT = "analysis/forecast_review_2026_11/data"
os.makedirs(OUT, exist_ok=True)
START = "2026-10-10"

pd.DataFrame(PriceHistory.objects.filter(date_time__gte=START).values("date_time", "day_ahead", "agile")).to_csv(f"{OUT}/prices.csv", index=False)
fcs = pd.DataFrame(Forecasts.objects.filter(created_at__gte=START).values("id", "name", "created_at"))
fcs.to_csv(f"{OUT}/forecasts.csv", index=False)
fd = pd.DataFrame(ForecastData.objects.filter(forecast_id__in=list(fcs["id"])).values())
fd.drop(columns=["shap_top_features"]).to_csv(f"{OUT}/forecast_data.csv", index=False)
ext = pd.DataFrame(ExternalForecast.objects.filter(source_created_at__gte=START).values())
ext.to_csv(f"{OUT}/external.csv", index=False)
print(len(fcs), len(fd), len(ext))
print(fcs.tail(8).to_string())
