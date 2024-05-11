# %%
import requests
import pandas as pd
from http import HTTPStatus
from requests.exceptions import HTTPError
from datetime import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

OCTOPUS_PRODUCT_URL = r"https://api.octopus.energy/v1/products/"
RETRIES = 3
RETRY_CODES = [
    HTTPStatus.TOO_MANY_REQUESTS,
    HTTPStatus.INTERNAL_SERVER_ERROR,
    HTTPStatus.BAD_GATEWAY,
    HTTPStatus.SERVICE_UNAVAILABLE,
    HTTPStatus.GATEWAY_TIMEOUT,
]

regions = {
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


def day_ahead_to_agile(df, reverse=False, region="G"):
    df.index = df.index.tz_convert("GB")
    x = pd.DataFrame(df).set_axis(["In"], axis=1)
    x["Out"] = x["In"]
    x["Peak"] = (x.index.hour >= 16) & (x.index.hour < 19)
    if reverse:
        x.loc[x["Peak"], "Out"] -= regions[region]["factors"][1]
        x["Out"] /= regions[region]["factors"][0]
    else:
        x["Out"] *= regions[region]["factors"][0]
        x.loc[x["Peak"], "Out"] += regions[region]["factors"][1]

    if reverse:
        name = "day_ahead"
    else:
        name = "agile"

    return x["Out"].rename(name)


def get_agile(start=pd.Timestamp("2023-07-01"), tz="GB", region="G"):
    start = pd.Timestamp(start).tz_convert("UTC")
    product = "AGILE-22-08-31"
    df = pd.DataFrame()
    url = f"{OCTOPUS_PRODUCT_URL}{product}"

    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta("48h")
    code = f"E-1R-{product}-{region}"
    url = url + f"/electricity-tariffs/{code}/standard-unit-rates/"

    x = []
    while end > start:
        print(start, end)
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


def get_nordpool(start):
    url = "https://www.nordpoolgroup.com/api/marketdata/page/325?currency=GBP"

    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

    except requests.exceptions.RequestException as e:
        return

    index = []
    data = []
    for row in r.json()["data"]["Rows"]:
        for column in row:
            if isinstance(row[column], list):
                for i in row[column]:
                    if i["CombinedName"] == "CET/CEST time":
                        if len(i["Value"]) > 10:
                            time = f"T{i['Value'][:2]}:00"
                            # print(time)
                    else:
                        if len(i["Name"]) > 8:
                            try:
                                # self.log(time, i["Name"], i["Value"])
                                data.append(float(i["Value"].replace(",", ".")))
                                index.append(
                                    pd.Timestamp(
                                        i["Name"].split("-")[2]
                                        + "-"
                                        + i["Name"].split("-")[1]
                                        + "-"
                                        + i["Name"].split("-")[0]
                                        + " "
                                        + time
                                    )
                                )
                            except:
                                pass

    price = pd.Series(index=index, data=data).sort_index()
    price.index = price.index.tz_localize("CET")
    price.index = price.index.tz_convert("GB")
    price = price[~price.index.duplicated()]
    return price.loc[start:]


def get_eex():
    url = "https://www.epexspot.com/en/market-data?market_area=GB&trading_date=2024-05-04&delivery_date=2024-05-05&underlying_year=&modality=Auction&sub_modality=DayAhead&technology=&product=60&data_mode=table&period=&production_period="
    try:
        r = requests.get(url)
        r.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

    except requests.exceptions.RequestException as e:
        return

    soup = BeautifulSoup(r.text, "html.parser")

    data = []
    table = soup.find("table")
    table_body = table.find("tbody")

    rows = table_body.find_all("tr")
    for row in rows:
        cols = row.find_all("td")
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele])  # Get rid of empty values

    lines = soup.get_text().split("\n")
    index = pd.date_range(
        pd.Timestamp(lines[[lines.index(l) for l in lines if "CET" in l][0] - 1].strip()) + pd.Timedelta("24h"),
        tz="CET",
        periods=24,
        freq="1h",
    ).tz_convert('GB')
    return pd.DataFrame(index=index, data=data)[3].astype(float)


# %%
fig, ax = plt.subplots(1, 1, figsize=(16, 6), layout="tight")
nd = get_nordpool("2024-01-01")
nd.plot(ax=ax)
eex = get_eex()
eex.plot(ax=ax)
da = day_ahead_to_agile(get_agile(nd.index[0]), reverse=True)
da.plot(ax=ax)

# %%
ax = eex.plot()
nd.loc[eex.index[0]:].plot(ax=ax)
da.loc[eex.index[0]:].plot(ax=ax)

# %%
