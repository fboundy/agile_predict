# %%
import requests
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import numpy as np

from http import HTTPStatus
from requests.exceptions import HTTPError
from urllib import parse

from datetime import datetime

OCTOPUS_PRODUCT_URL = r"https://api.octopus.energy/v1/products/"

TIME_FORMAT = "%d/%m %H:%M %Z"
MAX_ITERS = 3
RETRIES = 3
RETRY_CODES = [
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
]


def _rsq(x, y, order=1):
    # x=x.to_numpy()
    # y=y.to_numpy()
    coefficients = np.polyfit(x, y, order)
    polynomial = np.poly1d(coefficients)

    # Predicted values
    y_pred = polynomial(x)

    # Compute R^2
    ss_res = np.sum((y - y_pred) ** 2)  # Residual sum of squares
    ss_tot = np.sum((y - np.mean(y)) ** 2)  # Total sum of squares
    return 1 - (ss_res / ss_tot), y_pred


regions = {
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


def get_agile(start=pd.Timestamp("2023-07-01", tz="GB"), tz="GB", region="G"):
    if isinstance(start, pd.Timestamp) and start.tzinfo is not None:
        start = pd.Timestamp(start).tz_convert("UTC")
    else:
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
        x.loc[x["Peak"], "Out"] -= regions[region]["factors"][1]
        x["Out"] /= regions[region]["factors"][0]
    else:
        # print(region)
        x["Out"] *= regions[region]["factors"][0]
        x.loc[x["Peak"], "Out"] += regions[region]["factors"][1]

    if reverse:
        name = "day_ahead"
    else:
        name = "agile"

    return x["Out"].rename(name)


# %%
neso_url = "https://api.neso.energy/api/3/action/datastore_search"
df = {}

# %%
gen_mix = {"resource_id": "f93d1835-75bc-43e5-84ad-12472b180a98", "limit": 300000}
r = requests.get(neso_url, params=gen_mix)
df = pd.json_normalize(r.json(), record_path=["result", "records"]).set_index("DATETIME").sort_index()
df.index = pd.to_datetime(df.index)
df = df.loc["2023":]
# %%
# %%
demand = {
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
}

dfd = []
for year in demand:
    r = requests.get(neso_url, params={"resource_id": demand[year], "limit": 20000})
    dfd.append(pd.json_normalize(r.json(), record_path=["result", "records"]))
    dfd[-1].index = pd.date_range(f"{year}-01-01", freq="30min", periods=len(dfd[-1]), tz="GB")
    # dfd[-1] = dfd[-1][["ND", "TSD"]]
dfd = pd.concat(dfd).sort_index()
# %% demand update
params = {"resource_id": "177f6fa4-ae49-4182-81ea-0c6b35f26ca6", "limit": 5000}
r = requests.get(neso_url, params=params)
dfu = pd.json_normalize(r.json(), record_path=["result", "records"]).sort_values(
    ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]
)
dfu.index = pd.date_range(dfu["SETTLEMENT_DATE"].min(), periods=len(dfu), freq="30min", tz="GB")
dfu = dfu[dfu["FORECAST_ACTUAL_INDICATOR"] == "A"]
dfu = dfu.loc[dfd.index[-1] + pd.Timedelta("30min") :]
demand = pd.concat([dfd, dfu]).sort_index()
# #%%
# ax=dfu['ND'].plot(figsize=(16,6))
# dfd['ND'].loc[dfu.index[0]:].plot(ax=ax)
# # %%
# start_date = pd.Timestamp("2023-01-01")
# dfi = []
# while start_date < pd.Timestamp.now():
#     print(start_date)
#     end_date = min(start_date + pd.Timedelta("27D"), pd.Timestamp.now())
#     indo_url = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"
#     indo_params = {
#         "settlementDateFrom": start_date.strftime("%Y-%m-%d"),
#         "settlementDateTo": end_date.strftime("%Y-%m-%d"),
#     }

#     r = requests.get(indo_url, indo_params)
#     dfi.append(pd.json_normalize(r.json(), record_path=["data"]).set_index("startTime"))
#     dfi[-1].index = pd.to_datetime(dfi[-1].index)
#     start_date = start_date + pd.Timedelta("28D")
# dfi = pd.concat(dfi)[["initialDemandOutturn", "initialTransmissionSystemDemandOutturn"]].set_axis(
#     ["INDO", "ITSDO"], axis=1
# )
# %%
ag = get_agile(start="2023-01-01")
da = day_ahead_to_agile(ag, reverse=True)
# %%
ag_peak = ag[(ag.index.hour >= 16) & (ag.index.hour < 19)]
ag_off_peak = ag[(ag.index.hour < 16) | (ag.index.hour >= 19)]
ag_night = ag[(ag.index.hour >= 1) & (ag.index.hour < 5)]

res = []
for df, desc in zip(
    [ag, ag_peak, ag_off_peak, ag_night],
    ["All Prices", "Peak Prices (16:00 - 19:00)", "Off Peak Prices", "Night Prices (01:00-05:00)"],
):
    print(desc)
    print("         2023   2024")
    print("        -----  -----")
    for q in [x / 10 for x in range(1, 10)]:
        print(f"{q*100:4.0f}%: {df.loc['2023-11'].quantile(q):6.2f} {df.loc['2024-11'].quantile(q):6.2f}")
    print("        -----  -----")
    print(f"Mean:  {df.loc['2023-11'].mean():6.2f} {df.loc['2024-11'].mean():6.2f}\n\n")

    idx = [x / 10 for x in range(1, 10)] + ["Mean"]
    data = {
        (desc, year): [df.loc[f"{year}-11"].quantile(q).round(1) for q in [x / 10 for x in range(1, 10)]]
        + [df.loc[f"{year}-11"].mean().round(1)]
        for year in [2023, 2024]
    }
    res.append(pd.DataFrame(index=idx, data=data))
# %%
years = [2023, 2024]
best_10 = pd.DataFrame(
    index=range(1, 31),
    data={year: [ag.loc[f"{year}-11-{i+1}"].sort_values().iloc[:10].mean() for i in range(30)] for year in years},
)

# %%
res = pd.concat(res, axis=1)
res.to_csv("C:\\temp\\res.csv")
# %%
r2 = pd.DataFrame(
    index=pd.date_range("2023-01-01", periods=23, freq="1MS"),
    data={f"{dem}_{gen}": 0 for dem in ["ND", "TSD"] for gen in ["LOWC", "WIND"]},
)
# %%
for month in range(7, 12):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharey=True)
    fig1, ax1 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, layout="tight")
    fig2, ax2 = plt.subplots(1, 2, figsize=(10.6, 6), sharex=True)
    for year in 2023, 2024:
        date = pd.Timestamp(f"{year}-{month}-01")
        i = year - 2023
        start = f"{year}-{month}"
        x = pd.concat(
            [demand.loc[start], df.loc[start][["WIND", "SOLAR", "NUCLEAR"]], da.loc[start].rename("DAY_AHEAD")], axis=1
        )
        x["WIND"] = x["WIND"] - x["EMBEDDED_WIND_GENERATION"]
        (x["ND"] / 1000).plot(ax=ax[i], label="ND")
        # (dfd["ND"] / 1000).loc[start].plot(ax=ax[i])
        x["LOWC"] = x[["NUCLEAR", "WIND"]].sum(axis=1)
        # ax[i].plot(x["LOWC"]/1000, color='black', lw=1)
        axx = ax[i].twinx()
        x["DAY_AHEAD"].plot(ax=axx, color="red", label="Day Ahead Price")
        (x[["NUCLEAR", "WIND"]] / 1000).plot.area(ax=ax[i], stacked=True, color=["grey", "green"], lw=0)

        for row, dem in enumerate(["ND", "TSD"]):
            for col, gen in enumerate(["LOWC", "WIND"]):
                x["DIFF"] = (x[dem] - x[gen]) / 1000
                try:
                    rsq = _rsq(x["DIFF"], x["DAY_AHEAD"], degree=6)
                except:
                    rsq = np.nan
                sns.regplot(
                    x=x["DIFF"],
                    y=x["DAY_AHEAD"],
                    order=6,
                    ax=ax1[row][col],
                    label=f"{date.strftime('%b %Y')}: {rsq:0.3f}",
                    scatter_kws={"alpha": 0.2, "s": 5},
                )
                ax1[row][col].legend()

        x["DAY_AHEAD"].plot.hist(bins=50, ax=ax2[0], color=f"C{i}", alpha=0.5)
        ax2[1].plot(x["DAY_AHEAD"].sort_values(), [i / len(x) for i in range(len(x))])

        axx.set_ylim(-50, 400)
        ax[i].set_ylim(0, 45)
        ax[i].set_ylabel("Generation / Demand [GW]")
        axx.set_ylabel("Day Ahead Price (£/MWh)")
        # sns.kdeplot(x=x['DIFF'], y=x['DAY_AHEAD'], fill=True,ax=ax2[i])

    ax[1].legend(loc="upper right", bbox_to_anchor=(0.7, -0.05), ncols=4)
    axx.legend(loc="upper left", bbox_to_anchor=(0.7, -0.05))
    ax[0].get_legend().remove()
    for row, dem in enumerate(["National", "Transmission"]):
        ax1[row][0].set_ylabel("Day Ahead Price [£/MWh]")
        ax1[row][1].set_ylabel("")
        for col, gen in enumerate(["Low Carbon", "Renewable"]):
            if row == 1:
                ax1[row][col].set_xlabel(f"Corrected Demand minus Generation[GW]", fontsize=10)
            else:
                ax1[row][col].set_xlabel("")
            ax1[row][col].set_title(f"{dem} Demand | {gen} Generation")

    ax2[0].set_xlabel("Day Ahead Price [£/MWh]")
    ax2[1].set_ylim(0, 1)
    ax2[1].set_ylabel("Cumulative Probability")
    ax2[1].set_xlabel("Day Ahead Price [£/MWh]")
# ax[1].legend(bbox=(0.5,0,0.5,0.2))
# %%
fig1, ax1 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True, layout="tight")
r2 = pd.DataFrame(
    index=pd.date_range("2023-07-01", periods=17, freq="1MS"),
    data={f"{dem}_{gen}": 0.0 for dem in ["ND", "TSD"] for gen in ["LOWC", "WIND"]},
)

order = 4
for date in pd.date_range("2023-07-01", "2024-11-01", freq="1MS"):
    # for date in pd.date_range('2024-03-01', '2024-03-01', freq="1MS"):
    start = date.strftime("%Y-%m")
    print(start)
    x = pd.concat(
        [demand.loc[start], df.loc[start][["WIND", "SOLAR", "NUCLEAR"]], da.loc[start].rename("DAY_AHEAD")], axis=1
    )
    x["WIND"] = x["WIND"] - x["EMBEDDED_WIND_GENERATION"]
    x["LOWC"] = x[["NUCLEAR", "WIND"]].sum(axis=1)
    x = x[["WIND", "LOWC", "ND", "TSD", "DAY_AHEAD"]].dropna()
    for row, dem in enumerate(["ND", "TSD"]):
        for col, gen in enumerate(["LOWC", "WIND"]):
            x["DIFF"] = (x[dem] - x[gen]) / 1000
            rsq, pred = _rsq(x["DIFF"], x["DAY_AHEAD"], order=order)
            sns.regplot(
                x=x["DIFF"],
                y=x["DAY_AHEAD"],
                order=order,
                ax=ax1[row][col],
                label=f"{date.strftime('%b %Y')}: {rsq:0.3f}",
                scatter_kws={"alpha": 0.2, "s": 5},
            )
            ax1[row][col].legend()
            r2.loc[date, f"{dem}_{gen}"] = rsq

for ax in ax1.flatten():
    ax.legend(loc="upper left", ncols=2, fontsize=8)
# %%
start = pd.Timestamp("2024-11-01")
x = {
    "url": "https://api.nationalgrideso.com/api/3/action/datastore_search_sql",
    "params": parse.urlencode(
        {
            "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "f6d02c0f-957b-48cb-82ee-09003f2ba759" WHERE "SETTLEMENT_DATE" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}T00:00:00Z' ORDER BY "_id" ASC LIMIT 20000"""
        }
    ),
    "cols": [
        "IFA_FLOW",
        "TSD",
        "VIKING_FLOW",
        "IFA2_FLOW",
        "EMBEDDED_WIND_GENERATION",
        "ND",
        "MOYLE_FLOW",
        "NEMO_FLOW",
        "ELECLINK_FLOW",
        "PUMP_STORAGE_PUMPING",
        "EMBEDDED_WIND_CAPACITY",
        "ENGLAND_WALES_DEMAND",
        "EMBEDDED_SOLAR_CAPACITY",
        "SCOTTISH_TRANSFER",
        "NON_BM_STOR",
        "SETTLEMENT_PERIOD",
        "EAST_WEST_FLOW",
        "NSL_FLOW",
        "BRITNED_FLOW",
        "EMBEDDED_SOLAR_GENERATION",
    ],
    "record_path": ["result", "records"],
    "date_col": "SETTLEMENT_DATE",
    "period_col": "SETTLEMENT_PERIOD",
}

r = requests.get(x["url"], x["params"])
# %%

# %%
nationaldemand = {
    "url": "https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=7c0411cd-2714-4bb5-a408-adb065edf34d&limit=5000",
    "record_path": ["result", "records"],
    "date_col": "GDATETIME",
    "tz": "UTC",
    "cols": ["NATIONALDEMAND"],
}
r = requests.get(nationaldemand["url"])
nationaldemand_df = pd.json_normalize(r.json(), record_path=nationaldemand["record_path"])
nationaldemand_df = nationaldemand_df.set_index("GDATETIME")
nationaldemand_df.index = pd.to_datetime(nationaldemand_df.index)
# %%
ndf_from = "2024-12-08"
ndf_to = "2024-12-20"

ndf = {
    # "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF?publishDateTimeFrom={ndf_from}&publishDateTimeTo={ndf_to}",
    "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF",
    "params": {"publishDateTimeFrom": ndf_from, "publishDateTimeTo": ndf_to},
    "record_path": ["data"],
    "date_col": "startTime",
    "cols": "demand",
    "sort_col": "publishTime",
}
r = requests.get(ndf["url"])
ndf_df = pd.json_normalize(r.json(), record_path=ndf["record_path"])
ndf_df = ndf_df.set_index("startTime")
ndf_df.index = pd.to_datetime(ndf_df.index)
# %%
tsdf = {
    # "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF?publishDateTimeFrom={ndf_from}&publishDateTimeTo={ndf_to}",
    "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/TSDF",
    "params": {"publishDateTimeFrom": ndf_from, "publishDateTimeTo": ndf_to},
    "record_path": ["data"],
    "date_col": "startTime",
    "cols": "demand",
    "sort_col": "publishTime",
}
r = requests.get(tsdf["url"])
tsdf_df = pd.json_normalize(r.json(), record_path=ndf["record_path"])
tsdf_df = tsdf_df.set_index("startTime")
tsdf_df.index = pd.to_datetime(tsdf_df.index)
# %%
ndfd = {
    "url": "https://data.elexon.co.uk/bmrs/api/v1/forecast/demand/daily",
    "record_path": ["data"],
}

r = requests.get(ndfd["url"])
ndfd_df = pd.json_normalize(r.json(), record_path=ndfd["record_path"]).set_index("forecastDate")
ndfd_df.index = pd.to_datetime(ndfd_df.index)
ndfd_df = ndfd_df.resample("30min").ffill()
# %%
ax = ndfd_df[["nationalDemand", "transmissionSystemDemand"]].plot()
nationaldemand_df["NATIONALDEMAND"].plot(ax=ax)
ndf_df["demand"].plot(ax=ax)
tsdf_df[tsdf_df["boundary"] == "N"]["demand"].plot(ax=ax)

# %%
