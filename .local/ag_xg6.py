# %%
import pandas as pd
import requests
from urllib import parse
import matplotlib.pyplot as plt
from datetime import datetime

OCTOPUS_PRODUCT_URL = r"https://api.octopus.energy/v1/products/"
REGIONS = {
    "X": {
        "name": "National Average",
        "factors": (0.2136, 12.21),
    },
    "A": {
        "name": "Eastern England",
        "factors": (0.21, 13),
    },
    "B": {
        "name": "East Midlands",
        "factors": (0.20, 14),
    },
    "C": {
        "name": "London",
        "factors": (0.20, 12),
    },
    "D": {
        "name": "Merseyside and Northern Wales",
        "factors": (0.22, 13),
    },
    "E": {
        "name": "West Midlands",
        "factors": (0.21, 11),
    },
    "F": {
        "name": "North Eastern England",
        "factors": (0.21, 12),
    },
    "G": {
        "name": "North Western England",
        "factors": (0.21, 12),
    },
    "H": {
        "name": "Southern England",
        "factors": (0.21, 12),
    },
    "J": {
        "name": "South Eastern England",
        "factors": (0.22, 12),
    },
    "K": {
        "name": "Southern Wales",
        "factors": (0.22, 12),
    },
    "L": {
        "name": "South Western England",
        "factors": (0.23, 11),
    },
    "M": {
        "name": "Yorkshire",
        "factors": (0.20, 13),
    },
    "N": {
        "name": "Southern Scotland",
        "factors": (0.21, 13),
    },
    "P": {
        "name": "Northern Scotland",
        "factors": (0.24, 12),
    },
}


def _oct_time(d):
    # print(d)
    return datetime(
        year=pd.Timestamp(d).year,
        month=pd.Timestamp(d).month,
        day=pd.Timestamp(d).day,
    )


def get_agile(start=pd.Timestamp("2023-07-01"), tz="GB", region="G"):
    try:
        start = pd.Timestamp(start).tz_convert("UTC")
    except:
        start = pd.Timestamp(start).tz_localize("UTC")

    product = "AGILE-22-08-31"
    df = pd.DataFrame()
    url = f"{OCTOPUS_PRODUCT_URL}{product}"

    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta("48h")
    code = f"E-1R-{product}-{region}"
    url = url + f"/electricity-tariffs/{code}/standard-unit-rates/"

    x = []
    while end > start:
        # print(start, end)
        params = {
            "page_size": 1500,
            "order_by": "period",
            "period_from": _oct_time(start),
            "period_to": _oct_time(end),
        }

        r = requests.get(url, params=params)
        if "results" in r.json():
            x = x + r.json()["results"]
        end = pd.Timestamp(x[-1]["valid_from"]).ceil("24h")

    df = pd.DataFrame(x).set_index("valid_from")[["value_inc_vat"]]
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_convert(tz)
    df = df.sort_index()["value_inc_vat"]
    df = df[~df.index.duplicated()]
    return df.rename("agile")


def day_ahead_to_agile(df, reverse=False, region="G"):
    df.index = df.index.tz_convert("GB")
    x = pd.DataFrame(df).set_axis(["In"], axis=1)
    x["Out"] = x["In"]
    x["Peak"] = (x.index.hour >= 16) & (x.index.hour < 19)
    if reverse:
        x.loc[x["Peak"], "Out"] -= REGIONS[region]["factors"][1]
        x["Out"] /= REGIONS[region]["factors"][0]
    else:
        # print(region)
        x["Out"] *= REGIONS[region]["factors"][0]
        x.loc[x["Peak"], "Out"] += REGIONS[region]["factors"][1]

    if reverse:
        name = "day_ahead"
    else:
        name = "agile"

    return x["Out"].rename(name)


# %%
x = {
    "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/FOU2T14D?format=json",
    "record_path": ["data"],
}

dfs = []
for d in pd.date_range("2024-07-01", "2024-10-22"):
    r = requests.get(x["url"], {"publishDate": d.strftime("%Y-%m-%d")})
    dfs += [pd.json_normalize(r.json(), x["record_path"])]

df = pd.concat(dfs)
df.index = pd.to_datetime(df["forecastDate"])
# %%
fuels = list(df["fuelType"].drop_duplicates())
pTimes = list(df["publishTime"].drop_duplicates())
for fuel in fuels:
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(fuel)
    dff = df[df["fuelType"] == fuel]
    for pTime in pTimes:
        dff[dff["publishTime"] == pTime].plot(y="outputUsable", ax=ax, legend=False)

# %%
gen = []
dem = []
for d in pd.date_range("2024-07-01", "2024-10-22", freq="7D"):
    x = {
        "gen_url": "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELINST",
        "dem_url": "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn",
        "record_path": ["data"],
        "params": {
            "settlementDateFrom": d.strftime("%Y-%m-%d"),
            "settlementDateTo": (d + pd.Timedelta("6D")).strftime("%Y-%m-%d"),
        },
    }
    r = requests.get(x["gen_url"], x["params"])
    gen += [pd.json_normalize(r.json(), record_path=x["record_path"])[["startTime", "fuelType", "generation"]]]
    r = requests.get(x["dem_url"], x["params"])
    dem += [
        pd.json_normalize(r.json(), record_path=x["record_path"])[
            ["startTime", "initialDemandOutturn", "initialTransmissionSystemDemandOutturn"]
        ]
    ]

# %%
x = pd.concat(gen)
y = x.set_index(["startTime", "fuelType"]).unstack()
y.index = pd.to_datetime(y.index)
y = y.resample("30min").mean()
# %%
z = pd.concat(dem).set_index("startTime")
z.index = pd.to_datetime(z.index)
z.columns = pd.MultiIndex.from_tuples([("demand", c) for c in z.columns])
x = pd.concat([y, z], axis=1).loc["2024-10"]
ag = get_agile()
# %%
fig, ax = plt.subplots(figsize=(16, 6))
x["demand"]["initialDemandOutturn"].plot(ax=ax, lw=3, color="black")
x["demand"]["initialTransmissionSystemDemandOutturn"].plot(ax=ax, lw=2, ls="--", color="black")
# x['generation'][['NUCLEAR', 'WIND']].clip(0).plot.area(stacked=True,ax=ax)
x["generation"].clip(0).plot.area(stacked=True, ax=ax)

ax2 = ax.twinx()
x[("price", "agile")] = ag.loc["2024-10"]
x[("price", "day_ahead")] = day_ahead_to_agile(x[("price", "agile")], reverse=True)
x[("price", "day_ahead")].plot(ax=ax2, color="green")
# %%
fig, ax = plt.subplots()
for i, demand in enumerate(x["demand"].columns):
    x["diff"] = x["demand"][demand] - x["generation"][["NUCLEAR", "WIND"]].sum(axis=1)
    x.plot.scatter(x="diff", y=("price", "day_ahead"), ax=ax, alpha=0.5, color=f"C{i}")
# %%
fig, ax = plt.subplots()
x["demand"].plot.scatter(x=0, y=1, ax=ax, c=x["generation"]["WIND"])
# %%
