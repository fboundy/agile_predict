# %%
import pandas as pd
import requests

x = {
    "url": "https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=93c3048e-1dab-4057-a2a9-417540583929&limit=1000",
    "record_path": ["result", "records"],
    "tz": "GB",
    "date_col": "Datetime",
    "cols": ["Wind_Forecast"],
    "rename": ["bm_wind"],
}

y = {
    "url": "https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=b2f03146-f05d-4824-a663-3a4f36090c71&limit=1000",
    "record_path": ["result", "records"],
    "tz": "GB",
    "date_col": "Datetime",
    "cols": ["Wind_Forecast"],
    "rename": ["bm_wind"],
}

# %%
r = requests.get(x["url"])
df = pd.json_normalize(r.json(), record_path=x["record_path"])
df.index = pd.to_datetime(df["Datetime"])
df.index = df.index.tz_localize("UTC")

# %%
r = requests.get(y["url"])
dfy = pd.json_normalize(r.json(), record_path=x["record_path"])
dfy.index = pd.to_datetime(dfy["Datetime_GMT"])
dfy.index = dfy.index.tz_localize("UTC")
# %%
ax = dfy["Incentive_forecast"].plot(lw=3)
df["Wind_Forecast"].iloc[:100].plot(ax=ax, lw=1)
# %%
