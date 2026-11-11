"""Half-hourly ACTUALS panel, Jul 2023 -> now: settled price, NESO generation mix, TTF, continental wind.

Three winters of history for the structural tests (stack.py, ceiling.py, longhist.py).
Run: .venv/bin/python manage.py shell < analysis/forecast_review_2026_11/panel.py
"""
import pandas as pd
import requests

from config.utils import get_gas_ttf_history, get_open_meteo_eu_wind
from prices.models import PriceHistory

D = "analysis/forecast_review_2026_11/data"
START = "2023-07-01"

price = pd.DataFrame(PriceHistory.objects.values("date_time", "day_ahead")).set_index("date_time")["day_ahead"]

# NESO historic generation mix. GENERATION = all fuels + IMPORTS + WIND_EMB + SOLAR (checked on one row).
frames = []
for a in pd.date_range(START, pd.Timestamp.now(), freq="182D"):
    b = a + pd.Timedelta(days=182)
    sql = (f'SELECT "DATETIME","GAS","COAL","NUCLEAR","WIND","WIND_EMB","HYDRO","IMPORTS","BIOMASS","OTHER","SOLAR","STORAGE","GENERATION" '
           f'FROM "f93d1835-75bc-43e5-84ad-12472b180a98" WHERE "DATETIME" >= \'{a:%Y-%m-%d}\' AND "DATETIME" < \'{b:%Y-%m-%d}\' ORDER BY "DATETIME" LIMIT 20000')
    r = requests.get("https://api.neso.energy/api/3/action/datastore_search_sql", params={"sql": sql}, timeout=120).json()["result"]["records"]
    frames.append(pd.DataFrame(r))
    print(a.date(), len(r))
mix = pd.concat(frames)
mix["DATETIME"] = pd.to_datetime(mix["DATETIME"], utc=True)
mix = mix.set_index("DATETIME").astype(float).sort_index()
mix = mix[~mix.index.duplicated()]
mix.columns = mix.columns.str.lower()

p = mix.join(price, how="inner")
p["wind"] = p["wind"] + p["wind_emb"]
# Residual demand: what gas, imports, biomass, hydro and storage had to supply.
p["resid"] = p["generation"] - p["wind"] - p["solar"] - p["nuclear"]

gas = get_gas_ttf_history(START)
gas.index = gas.index.tz_convert("UTC")
p["gas_ttf"] = gas.reindex(p.index, method="ffill").bfill()

eu = []
for a in pd.date_range(START, pd.Timestamp.now(), freq="365D"):
    b = min(a + pd.Timedelta(days=364), pd.Timestamp.now().normalize() - pd.Timedelta(days=1))
    s = get_open_meteo_eu_wind(url="https://historical-forecast-api.open-meteo.com/v1/forecast", start_date=f"{a:%Y-%m-%d}", end_date=f"{b:%Y-%m-%d}")
    print("eu", a.date(), len(s))
    eu.append(s)
eu = pd.concat(eu)
p["eu_wind_cf"] = eu[~eu.index.duplicated()].reindex(p.index)

p.to_pickle(f"{D}/panel.pkl")
print(p.shape, p.index.min(), p.index.max())
print(p.isna().mean().round(3).to_string())
print(p.resample("QS").agg({"day_ahead": "mean", "gas_ttf": "mean", "generation": "mean", "resid": ["mean", "max"], "imports": ["min", "max"], "eu_wind_cf": "count"}).round(1).to_string())
