import requests
import time
import pandas as pd
import xgboost as xg

from http import HTTPStatus
from requests.exceptions import HTTPError
from urllib import parse
from datetime import datetime

from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData

import matplotlib.pyplot as plt

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

AGILE_FACTORS = {
    "import": {
        "A": (0.21, 0, 13),
        "B": (0.20, 0, 14),
        "C": (0.20, 0, 12),
        "D": (0.22, 0, 13),
        "E": (0.21, 0, 12),
        "F": (0.21, 0, 12),
        "G": (0.21, 0, 12),
        "H": (0.21, 0, 12),
        "J": (0.22, 0, 12),
        "K": (0.22, 0, 12),
        "L": (0.23, 0, 11),
        "M": (0.20, 0, 13),
        "N": (0.21, 0, 13),
        "P": (0.24, 0, 12),
    },
    "export": {
        "A": (0.095, 1.09, 7.04),
        "B": (0.094, 0.78, 6.27),
        "C": (0.095, 1.30, 5.93),
        "D": (0.097, 1.26, 5.97),
        "E": (0.094, 0.77, 6.50),
        "F": (0.095, 0.87, 4.88),
        "G": (0.096, 1.10, 5.89),
        "H": (0.094, 0.93, 7.05),
        "J": (0.094, 1.09, 7.41),
        "K": (0.094, 0.97, 5.46),
        "L": (0.093, 0.83, 7.14),
        "M": (0.096, 0.72, 5.78),
        "N": (0.097, 0.90, 3.85),
        "P": (0.096, 1.36, 2.68),
    },
}


def _oct_time(d):
    # print(d)
    return datetime(
        year=pd.Timestamp(d).year,
        month=pd.Timestamp(d).month,
        day=pd.Timestamp(d).day,
    )


def get_history_from_model():
    if History.objects.count() == 0:
        df = pd.DataFrame()
    else:
        df = pd.DataFrame(list(History.objects.all().values()))
        df["time"] = df["date_time"].dt.hour + df["date_time"].dt.minute / 60
        df["day_of_week"] = df["date_time"].dt.day_of_week.astype(int)
        df["day_of_year"] = df["date_time"].dt.day_of_year.astype(int)
        df.index = pd.to_datetime(df["date_time"])
        df.index = df.index.tz_convert("GB")
        df.drop(["id", "date_time"], axis=1, inplace=True)

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
                "publishDateTimeTo": pd.Timestamp.now().strftime("%Y-%m-%d"),
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
            "url": "https://api.nationalgrideso.com/api/3/action/datastore_search?resource_id=f93d1835-75bc-43e5-84ad-12472b180a98&limit=1000000&sort=DATETIME",
            "params": {"offset": 254110 + delta},
            "record_path": ["result", "records"],
            "date_col": "DATETIME",
            "cols": ["SOLAR"],
            "rename": ["solar"],
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
    ]

    hist = pd.concat([DataSet(**x).download() for x in history_data], axis=1)
    print(hist)

    if isinstance(hist["ND"], pd.DataFrame):
        hist["demand"] = hist["ND"].mean(axis=1)
    else:
        hist["demand"] = hist["ND"]
    hist.index = pd.to_datetime(hist.index)
    hist = hist.drop("ND", axis=1).sort_index()

    return hist.astype(float).dropna()


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
            "cols": ["EMBEDDED_SOLAR_FORECAST"],
            "rename": ["solar"],
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

    df = pd.concat([DataSet(**x).download() for x in forecast_data], axis=1)
    df["demand"] = df[["demand", "NATIONALDEMAND"]].mean(axis=1)
    df["date_time"] = df.index
    df["time"] = df["date_time"].dt.hour + df["date_time"].dt.minute / 60
    df["day_of_week"] = df["date_time"].dt.day_of_week.astype(int)
    df["day_of_year"] = df["date_time"].dt.day_of_year.astype(int)

    df.index = df.index.tz_convert("GB")
    df.drop(["date_time", "NATIONALDEMAND"], axis=1, inplace=True)

    return df.sort_index().dropna()


class DataSet:
    # _metadata = ["url", "hdf", "params", "date_col", "tz", "tz_out", "record_path"]

    # @property
    # def _constructor(self):
    #     return DataSet

    def __init__(self, *args, **kwargs) -> None:
        self.params = kwargs.pop("params", {})
        self.tz = kwargs.pop("tz", "UTC")
        self.__dict__ = self.__dict__ | kwargs
        # self.__dict__ = self.__dict__ | kwargs

    def update(self, download_all=False, hdf=None):
        pass

    def download(self, tz="GB", params={}):
        print(self.url)
        for n in range(RETRIES):
            try:
                response = requests.get(url=self.url, params=self.params)
                response.raise_for_status()
                break

            except HTTPError as exc:
                code = exc.response.status_code

                if code in RETRY_CODES:
                    # retry after n seconds
                    time.sleep(n)
                    continue
        # return response.json()
        try:
            df = pd.json_normalize(response.json(), self.record_path)
        except:
            try:
                df = pd.DataFrame(response.json()[self.record_path[0]])
            except:
                return response.json()

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
        return df


def get_agile(start=pd.Timestamp("2023-07-01"), tz="GB", area="G"):
    start = pd.Timestamp(start).tz_convert("UTC")
    product = "AGILE-22-08-31"
    df = pd.DataFrame()
    url = f"{OCTOPUS_PRODUCT_URL}{product}"

    end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta("48h")
    code = f"E-1R-{product}-{area}"
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


def day_ahead_to_agile(df, reverse=False, area="G"):
    x = pd.DataFrame(df).set_axis(["In"], axis=1)
    x["Out"] = x["In"]
    x["Peak"] = (x.index.hour >= 16) & (x.index.hour < 19)
    if reverse:
        x.loc[x["Peak"], "Out"] -= AGILE_FACTORS["import"][area][2]
        x["Out"] /= AGILE_FACTORS["import"][area][0]
    else:
        x["Out"] *= AGILE_FACTORS["import"][area][0]
        x.loc[x["Peak"], "Out"] += AGILE_FACTORS["import"][area][2]

    if reverse:
        name = "day_ahead"
    else:
        name = "agile"

    return x["Out"].rename(name)


def df_to_Model(df, myModel):
    for index, row in df.iterrows():
        x = {"date_time": index} | row.to_dict()
        obj = myModel(**x)
        obj.save()


class Command(BaseCommand):
    # def add_arguments(self, parser):
    #     parser.add_argument("poll_ids", nargs="+", type=int)

    def handle(self, *args, **options):
        # PriceHistory.objects.all().delete()

        print("Getting history from model")
        hist = get_history_from_model()

        print("Database History:")
        print(hist)
        start = pd.Timestamp("2023-07-01", tz="GB")
        if len(hist) > 0:
            start = hist.index[-1] + pd.Timedelta("30min")

        print(f"New data from {start.strftime(TIME_FORMAT)}:")

        new_hist = get_latest_history(start=start)
        if len(new_hist) > 0:
            print(new_hist)
            df_to_Model(new_hist, History)

        else:
            print("None")

        hist = pd.concat([hist, new_hist]).sort_index()

        print("Getting Historic Prices")
        prices = pd.DataFrame(list(PriceHistory.objects.all().values()))

        start = pd.Timestamp("2023-07-01", tz="GB")
        if len(prices) > 0:
            prices.index = pd.to_datetime(prices["date_time"])
            prices = prices.sort_index()
            prices.index = prices.index.tz_convert("GB")
            prices.drop(["id", "date_time"], axis=1, inplace=True)
            start = prices.index[-1] + pd.Timedelta("30min")

        agile = get_agile(start=start)
        day_ahead = day_ahead_to_agile(agile, reverse=True)

        new_prices = pd.concat([day_ahead, agile], axis=1)
        if len(prices) > 0:
            new_prices = new_prices[new_prices.index > prices.index[-1]]
        print(new_prices)
        if len(new_prices) > 0:
            print(new_prices)
            df_to_Model(new_prices, PriceHistory)

        prices = pd.concat([prices, new_prices]).sort_index()

        print("Getting latest Forecast")
        fc = get_latest_forecast()
        print(fc)

        # recent = pd.Timestamp.now(tz="GB") - pd.Timedelta("14d")
        # fig, ax = plt.subplots(1, 1, figsize=(16, 6), layout="tight")
        # prices["day_ahead"].loc[recent:].plot(ax=ax)
        # X = hist.drop("demand_source", axis=1)
        X = hist
        y = prices["day_ahead"].loc[hist.index]

        print(X)
        print(y)

        # model = LGBMRegressor(n_estimators=200, num_leaves=32)
        model = xg.XGBRegressor(objective="reg:squarederror", booster="dart")
        model.fit(X, y)
        # train_y = pd.Series(index=hist.index, data=model.predict(X))
        # train_y.loc[recent:].plot(ax=ax)

        cols = X.columns
        fc["day_ahead"] = model.predict(fc[cols])
        fc["agile_pred"] = day_ahead_to_agile(fc["day_ahead"]).astype(float).round(2)
        fc["agile_actual"] = prices["agile"].loc[fc.index[0] :]
        print(fc[["agile_pred", "agile_actual"]])
        fc.drop(["time", "day_of_week", "day_of_year", "day_ahead"], axis=1, inplace=True)

        f = Forecasts(name=pd.Timestamp.now(tz="GB").strftime("%Y-%m-%d %H:%M %z"))
        f.save()

        fc["forecast"] = f
        df_to_Model(fc, ForecastData)
