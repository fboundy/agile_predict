import pandas as pd
import requests
import time

from http import HTTPStatus
from requests.exceptions import HTTPError
from urllib import parse
from datetime import datetime
from config.settings import GLOBAL_SETTINGS
from django.core.management import call_command

from prices.models import History, PriceHistory, Forecasts, ForecastData, AgileData

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

regions = GLOBAL_SETTINGS["REGIONS"]


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
    return price.resample("30min").ffill().loc[start:]


def _oct_time(d):
    # print(d)
    return datetime(
        year=pd.Timestamp(d).year,
        month=pd.Timestamp(d).month,
        day=pd.Timestamp(d).day,
    )


def queryset_to_df(queryset):
    df = pd.DataFrame(list(queryset.values()))
    df["time"] = df["date_time"].dt.hour + df["date_time"].dt.minute / 60
    df["day_of_week"] = df["date_time"].dt.day_of_week.astype(int)
    # df["day_of_year"] = df["date_time"].dt.day_of_year.astype(int)
    df.index = pd.to_datetime(df["date_time"])
    df.index = df.index.tz_convert("GB")
    df.drop(["id", "date_time"], axis=1, inplace=True)

    return df


def get_history_from_model():
    if History.objects.count() == 0:
        df = pd.DataFrame()
    else:
        queryset = History.objects.all()
        df = queryset_to_df(queryset=queryset)

    return df.sort_index()


def get_forecast_from_model(forecast):
    if Forecasts.objects.count() == 0:
        df = pd.DataFrame()
    else:
        queryset = ForecastData.objects.filter(forecast=forecast)
        if queryset.count() > 0:
            df = queryset_to_df(queryset=queryset)
        else:
            df = pd.DataFrame()

    return df.sort_index()


def get_latest_history(start):
    delta = int((pd.Timestamp(start) - pd.Timestamp("2023-07-01", tz="GB")).total_seconds() / 1800)
    history_data = [
        {
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "bf5ab335-9b40-4ea4-b93a-ab4af7bce003" WHERE "SETTLEMENT_DATE" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}T00:00:00Z' ORDER BY "_id" ASC LIMIT 20000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "SETTLEMENT_DATE",
            "period_col": "SETTLEMENT_PERIOD",
            "cols": "ND",
        },
        {
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "f6d02c0f-957b-48cb-82ee-09003f2ba759" WHERE "SETTLEMENT_DATE" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}T00:00:00Z' ORDER BY "_id" ASC LIMIT 20000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "SETTLEMENT_DATE",
            "period_col": "SETTLEMENT_PERIOD",
            "cols": "ND",
        },
        {
            "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/INDO?format=json",
            "params": {
                "publishDateTimeFrom": (pd.Timestamp.now() - pd.Timedelta("27D")).strftime("%Y-%m-%d"),
                "publishDateTimeTo": (pd.Timestamp.now() + pd.Timedelta("1D")).strftime("%Y-%m-%d"),
            },
            "record_path": ["data"],
            "date_col": "startTime",
            "cols": ["demand"],
            "rename": ["ND"],
        },
        {
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "7524ec65-f782-4258-aaf8-5b926c17b966" WHERE "Datetime_GMT" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}T00:00:00Z' ORDER BY "_id" ASC LIMIT 40000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "Datetime_GMT",
            "tz": "UTC",
            "cols": ["Incentive_forecast"],
            "rename": ["bm_wind"],
        },
        {
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search_sql",
            "params": parse.urlencode(
                {
                    "sql": f"""SELECT COUNT(*) OVER () AS _count, * FROM "f93d1835-75bc-43e5-84ad-12472b180a98" WHERE "DATETIME" >= '{pd.Timestamp(start).strftime("%Y-%m-%d")}' ORDER BY "_id" ASC LIMIT 20000"""
                }
            ),
            "record_path": ["result", "records"],
            "date_col": "DATETIME",
            "cols": ["SOLAR", "WIND"],
            "rename": ["solar", "total_wind"],
        },
        {
            "url": "https://archive-api.open-meteo.com/v1/archive",
            "params": {
                "latitude": 54.0,
                "longitude": 2.3,
                "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp.now().normalize().strftime("%Y-%m-%d"),
                "hourly": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            },
            "record_path": ["hourly"],
            "date_col": "time",
            "tz": "UTC",
            "resample": "30min",
            "cols": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            "rename": ["temp_2m", "wind_10m", "rad"],
        },
        {
            "url": "https://api.open-meteo.com/v1/forecast",
            "params": {
                "latitude": 54.0,
                "longitude": 2.3,
                "start_date": (pd.Timestamp.now().normalize() - pd.Timedelta("5D")).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp.now().normalize().strftime("%Y-%m-%d"),
                "hourly": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            },
            "record_path": ["hourly"],
            "date_col": "time",
            "tz": "UTC",
            "resample": "30min",
            "cols": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            "rename": ["temp_2m_f", "wind_10m_f", "rad_f"],
        },
    ]

    downloaded_data = []
    download_errors = []

    for x in history_data:
        data, e = DataSet(**x).download()
        if len(data) > 0:
            downloaded_data += [data]
        else:
            download_errors += [e]

    hist = pd.concat(downloaded_data, axis=1).loc[: pd.Timestamp.now(tz="GB")]
    # print(hist.iloc[-48:].to_string())

    if isinstance(hist["ND"], pd.DataFrame):
        hist["demand"] = hist["ND"].mean(axis=1)
    else:
        hist["demand"] = hist["ND"]
    hist.index = pd.to_datetime(hist.index)
    hist = hist.drop("ND", axis=1).sort_index()

    meteo_cols = ["temp_2m", "wind_10m", "rad"]

    for c in [m for m in meteo_cols if m in hist.columns]:
        hist.loc[hist[c].isnull(), c] = hist.loc[hist[c].isnull(), f"{c}_f"]

    hist = hist.drop([f"{c}_f" for c in meteo_cols if c in hist.columns], axis=1)

    all_cols = ["total_wind", "bm_wind", "solar", "demand"] + meteo_cols
    missing_cols = [c for c in all_cols if c not in hist.columns]
    if len(missing_cols) > 0:
        print(f">>> ERROR: No historic data for {missing_cols} ")
        return pd.DataFrame(), missing_cols
    else:
        return hist.astype(float).dropna(), missing_cols


def get_latest_forecast():
    ndf_from = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    ndf_to = (pd.Timestamp.now().normalize() + pd.Timedelta("24h")).strftime("%Y-%m-%d")

    forecast_data = [
        {
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=93c3048e-1dab-4057-a2a9-417540583929&limit=1000",
            "record_path": ["result", "records"],
            "tz": "GB",
            "date_col": "Datetime",
            "cols": ["Wind_Forecast"],
            "rename": ["bm_wind"],
        },
        {
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=db6c038f-98af-4570-ab60-24d71ebd0ae5&limit=1000",
            "record_path": ["result", "records"],
            "tz": "UTC",
            "cols": ["EMBEDDED_SOLAR_FORECAST", "EMBEDDED_WIND_FORECAST"],
            "rename": ["solar", "emb_wind"],
            "date_col": "DATE_GMT",
            "time_col": "TIME_GMT",
        },
        {
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=7c0411cd-2714-4bb5-a408-adb065edf34d&limit=5000",
            "record_path": ["result", "records"],
            "date_col": "GDATETIME",
            "tz": "UTC",
            "cols": ["NATIONALDEMAND"],
        },
        {
            "url": "https://api.open-meteo.com/v1/forecast",
            "params": {
                "latitude": 54.0,
                "longitude": 2.3,
                "current": "temperature_2m",
                "minutely_15": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
                "forecast_days": 14,
            },
            "date_col": "time",
            "tz": "UTC",
            "resample": "30min",
            "record_path": ["minutely_15"],
            "cols": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
            "rename": ["temp_2m", "wind_10m", "rad"],
        },
        {
            # "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF?publishDateTimeFrom={ndf_from}&publishDateTimeTo={ndf_to}",
            "url": f"https://data.elexon.co.uk/bmrs/api/v1/datasets/NDF",
            "params": {"publishDateTimeFrom": ndf_from, "publishDateTimeTo": ndf_to},
            "record_path": ["data"],
            "date_col": "startTime",
            "cols": "demand",
            "sort_col": "publishTime",
        },
    ]

    downloaded_data = []
    download_errors = []

    for x in forecast_data:
        data, e = DataSet(**x).download()
        if len(data) > 0:
            downloaded_data += [data]
        else:
            download_errors += [e]

    df = pd.concat(downloaded_data, axis=1)

    demand_cols = ["demand", "NATIONALDEMAND"]
    if all([c in df.columns for c in demand_cols]):
        df["demand"] = df[demand_cols].mean(axis=1)
        df.drop(["NATIONALDEMAND"], axis=1, inplace=True)
        missing_cols = []
    elif "NATIONALDEMAND" not in df.columns:
        missing_cols = ["NATIONALDEMAND"]
    else:
        missing_cols = []

    all_cols = ["emb_wind", "bm_wind", "solar", "demand", "temp_2m", "wind_10m", "rad"]
    missing_cols += [c for c in all_cols if c not in df.columns]
    if len(missing_cols) > 0:
        print(f">>> ERROR: No forecast data for {missing_cols} ")
        return pd.DataFrame(), missing_cols
    else:
        df["date_time"] = pd.to_datetime(df.index)
        df["time"] = df["date_time"].dt.hour + df["date_time"].dt.minute / 60
        df["day_of_week"] = df["date_time"].dt.day_of_week.astype(int)
        # df["day_of_year"] = df["date_time"].dt.day_of_year.astype(int)

        df.index = pd.to_datetime(df.index).tz_convert("GB")
        df.drop(["date_time"], axis=1, inplace=True)

        return df.sort_index().dropna(), missing_cols


class DataSet:
    def __init__(self, *args, **kwargs) -> None:
        self.params = kwargs.pop("params", {})
        self.tz = kwargs.pop("tz", "UTC")
        self.__dict__ = self.__dict__ | kwargs
        # self.__dict__ = self.__dict__ | kwargs

    def update(self, download_all=False, hdf=None):
        pass

    def download(self, tz="GB", params={}):
        print(f"    {self.url}")
        for n in range(RETRIES):
            try:
                response = requests.get(url=self.url, params=self.params)
                response.raise_for_status()
                code = None
                break

            except HTTPError as exc:
                code = exc.response.status_code

                if code in RETRY_CODES:
                    # retry after n seconds
                    time.sleep(n)
                    continue

        try:
            df = pd.json_normalize(response.json(), self.record_path)
        except:
            try:
                df = pd.DataFrame(response.json()[self.record_path[0]])
            except Exception as e:
                print(f">>> ERROR {e} for URL {self.url}\n>>> with params {self.params}")
                return pd.DataFrame(), code

        try:
            df.index = pd.to_datetime(df[self.date_col])
            df.index = df.index.tz_localize(self.tz)
        except:
            pass

        try:
            df.index += pd.to_datetime(df[self.time_col], format="%H:%M") - pd.Timestamp("1900-01-01")
        except:
            pass

        try:
            df.index += (df[self.period_col] - 1) * pd.Timedelta("30min")
        except:
            pass

        try:
            df.index = df.index.tz_convert(tz)
        except:
            pass

        try:
            df = df[self.cols]
        except:
            pass

        try:
            df = df.resample(self.resample).mean()
        except:
            pass

        try:
            df = df.interpolate()
        except:
            pass

        try:
            df = df.sort_values(self.sort_col)
        except:
            pass

        if isinstance(df, pd.DataFrame):
            try:
                df = df.set_axis(self.rename, axis=1)
            except:
                pass
        elif isinstance(df, pd.Series):
            try:
                df = df.rename(self.rename)
            except:
                pass

        df = df.sort_index()
        df = df[~df.index.duplicated()]
        return df, None


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
        x["Out"] *= regions[region]["factors"][0]
        x.loc[x["Peak"], "Out"] += regions[region]["factors"][1]

    if reverse:
        name = "day_ahead"
    else:
        name = "agile"

    return x["Out"].rename(name)


def df_to_Model(df, myModel, update=False):
    df = df.dropna()
    for index, row in df.iterrows():
        if update:
            try:
                obj = myModel.objects.get(date_time=index)
                for key, value in row.items():
                    setattr(obj, key, value)
                obj.save()
            except myModel.DoesNotExist:
                new_values = {"date_time": index}
                new_values.update(row)
                obj = myModel(**new_values)
                obj.save()
        else:
            try:
                new_values = {"date_time": index}
                new_values.update(row)
                obj = myModel(**new_values)
                obj.save()
            except:
                print(f"Failed to update {myModel} with data for datetime {index}")


def model_to_df(myModel):
    df = pd.DataFrame(list(myModel.objects.all().values()))
    start = pd.Timestamp("2023-07-01", tz="GB")
    if len(df) > 0:
        df.index = pd.to_datetime(df["date_time"])
        df = df.sort_index()
        df.index = df.index.tz_convert("GB")
        df.drop(["id", "date_time"], axis=1, inplace=True)
        start = df.index[-1] + pd.Timedelta("30min")
    return df, start
