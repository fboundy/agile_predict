import pandas as pd
import numpy as np
import re
import requests
import time
import logging

from http import HTTPStatus
from requests.exceptions import HTTPError
from urllib import parse
from datetime import datetime
from config.settings import GLOBAL_SETTINGS
from django.core.management import call_command

from prices.models import History, PriceHistory, Forecasts, ForecastData, AgileData

OCTOPUS_PRODUCT_URL = r"https://api.octopus.energy/v1/products/"


logger = logging.getLogger(__name__)

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


def get_gb60():
    # url = "https://www.nordpoolgroup.com/api/marketdata/page/325?currency=GBP"
    url = "https://dataportal-api.nordpoolgroup.com/api/DayAheadPrices"

    params = {
        "date": (pd.Timestamp.now() + pd.Timedelta("13h")).strftime("%Y-%m-%d"),
        "market": "N2EX_DayAhead",
        "deliveryArea": "UK",
        "currency": "GBP",
    }

    try:
        r = requests.get(url, params=params)
        r.raise_for_status()  # Raise an exception for unsuccessful HTTP status codes

    except requests.exceptions.RequestException as e:
        return

    price = pd.Series(
        {
            pd.Timestamp(row["deliveryStart"]).tz_convert("GB"): float(row["entryPerArea"]["UK"])
            for row in r.json()["multiAreaEntries"]
        }
    )
    return price


def get_gas_ttf_history(start=None, end=None):
    def as_utc(value):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC")

    start = as_utc(start or "2023-07-01")
    end = as_utc(end or pd.Timestamp.now(tz="UTC")) + pd.Timedelta("1D")
    url = "https://query2.finance.yahoo.com/v8/finance/chart/TTF=F"
    params = {
        "period1": int(start.timestamp()),
        "period2": int(end.timestamp()),
        "interval": "1d",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except Exception:
        logger.exception("Unable to download Yahoo TTF=F gas prices")
        return pd.Series(dtype=float, name="gas_ttf")

    gas = pd.Series(
        data=closes,
        index=pd.to_datetime(timestamps, unit="s", utc=True),
        name="gas_ttf",
        dtype=float,
    ).dropna()
    gas.index = gas.index.tz_convert("GB")
    gas = gas.sort_index()
    return gas[(gas.index >= start.tz_convert("GB")) & (gas.index <= end.tz_convert("GB"))]


def gas_ttf_at(created_at, gas_history=None):
    created_at = pd.Timestamp(created_at)
    if created_at.tzinfo is None:
        created_at = created_at.tz_localize("UTC")
    created_at = created_at.tz_convert("GB")

    gas_history = gas_history if gas_history is not None else get_gas_ttf_history(end=created_at)
    if len(gas_history) == 0:
        return None

    gas = gas_history[gas_history.index <= created_at]
    if len(gas) == 0:
        return float(gas_history.iloc[0])
    return float(gas.iloc[-1])


def _availability_records_to_hh(records, col_name, start=None, end=None):
    """Broadcast daily availability records (one aggregated row per forecastDate) to 30-min GB slots."""
    if not records:
        return pd.Series(dtype=float, name=col_name)
    data = pd.DataFrame(records)
    data["forecastDate"] = pd.to_datetime(data["forecastDate"]).dt.tz_localize("GB")
    data = data.sort_values(["forecastDate", "publishTime"])
    data = data.drop_duplicates("forecastDate", keep="last")
    series = data.set_index("forecastDate")["outputUsable"].astype(float)

    def _as_gb(v):
        ts = pd.Timestamp(v)
        return ts.tz_localize("GB") if ts.tzinfo is None else ts.tz_convert("GB")

    s = _as_gb(start or series.index.min())
    e = _as_gb(end or (series.index.max() + pd.Timedelta("1D") - pd.Timedelta("30min")))
    series = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="1D", tz="GB")).ffill()
    hh = series.reindex(pd.date_range(s.floor("D"), e.ceil("D"), freq="30min", tz="GB")).ffill()
    return hh.loc[s:e].rename(col_name)


def nuclear_availability_to_half_hourly(df, start=None, end=None):
    return _availability_records_to_hh(df.to_dict("records") if len(df) else [], "nuclear", start, end)


def get_gas_availability_forecast(start=None, end=None):
    """
    Fetch CCGT + OCGT available capacity (MW) from the BMRS daily availability forecast.
    Same endpoint as nuclear, summed across both gas fuel types, 14 days ahead.
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/forecast/availability/daily"
    all_records = []
    for fuel in ("CCGT", "OCGT"):
        try:
            resp = requests.get(url, params={"fuelType": fuel, "format": "json"}, timeout=30)
            resp.raise_for_status()
            all_records.extend(resp.json().get("data", []))
        except Exception:
            logger.exception("Unable to download %s availability forecast", fuel)

    if not all_records:
        return pd.Series(dtype=float, name="gas_availability")

    # Sum CCGT + OCGT per forecastDate (one aggregated row per fuel per day from API)
    df = pd.DataFrame(all_records)
    df["outputUsable"] = pd.to_numeric(df["outputUsable"], errors="coerce")
    # Keep forecastDate as plain date string so _availability_records_to_hh can tz_localize it
    df["forecastDate"] = pd.to_datetime(df["forecastDate"]).dt.date.astype(str)
    combined = df.groupby("forecastDate")["outputUsable"].sum().reset_index()
    combined["publishTime"] = combined["forecastDate"]
    return _availability_records_to_hh(combined.to_dict("records"), "gas_availability", start, end)


def get_gas_availability_at(forecast_date, created_at):
    """Return the CCGT + OCGT available capacity (MW) that was published before created_at."""
    ts = pd.Timestamp(forecast_date)
    forecast_date = (ts.tz_localize("GB") if ts.tzinfo is None else ts.tz_convert("GB")).normalize()
    created_at    = pd.Timestamp(created_at).tz_convert("UTC")
    url = "https://data.elexon.co.uk/bmrs/api/v1/forecast/availability/daily/evolution"
    total = 0.0
    for fuel in ("CCGT", "OCGT"):
        try:
            resp = requests.get(
                url,
                params={"fuelType": fuel, "forecastDate": forecast_date.date().isoformat(), "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            records = resp.json().get("data", [])
            if not records:
                continue
            known = [r for r in records if pd.Timestamp(r["publishTime"], tz="UTC") <= created_at]
            src = known if known else records
            src_sorted = sorted(src, key=lambda r: r["publishTime"])
            total += float(src_sorted[-1]["outputUsable"])
        except Exception:
            logger.exception("Unable to fetch %s evolution for %s", fuel, forecast_date.date())
    return total if total > 0 else None


def get_open_meteo_fr_weather(start=None, end=None):
    """
    Fetch 16-day Open-Meteo weather forecast for central France (47°N, 2°E).
    Returns a DataFrame with columns fr_wind (m/s) and fr_rad (W/m²) at 30-min
    resolution, as a continental supply-side proxy for interconnector direction.
    High FR wind / solar → lower continental prices → more likely FR→GB flow.
    """
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":   47.0,
            "longitude":  2.0,
            "hourly":     "wind_speed_10m,shortwave_radiation",
            "wind_speed_unit": "ms",
            "timeformat": "unixtime",
            "timezone":   "UTC",
            "forecast_days": 16,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        times = pd.to_datetime(data["hourly"]["time"], unit="s", utc=True)
        df = pd.DataFrame({
            "fr_wind": data["hourly"]["wind_speed_10m"],
            "fr_rad":  data["hourly"]["shortwave_radiation"],
        }, index=times)
        df = df.resample("30min").interpolate(method="time")
        if start is not None and end is not None:
            idx = pd.date_range(
                start=pd.Timestamp(start).tz_convert("UTC"),
                end=pd.Timestamp(end).tz_convert("UTC"),
                freq="30min",
            )
            df = df.reindex(idx.union(df.index)).interpolate(method="time").reindex(idx)
        return df
    except Exception:
        logger.exception("Open-Meteo FR: failed to fetch French weather")
        return pd.DataFrame(columns=["fr_wind", "fr_rad"])


def _parse_entsoe_qty(xml_text):
    """Parse ENTSO-E generation XML into a UTC-indexed Series of quantities."""
    from xml.etree import ElementTree as ET
    root = ET.fromstring(xml_text)
    if "Acknowledgement" in root.tag:
        return pd.Series(dtype=float)
    records = []
    for ts in root.findall(".//{*}TimeSeries"):
        for period in ts.findall("{*}Period"):
            start_el = period.find("{*}timeInterval/{*}start")
            res_el   = period.find("{*}resolution")
            if start_el is None or res_el is None:
                continue
            start_ts = pd.Timestamp(start_el.text, tz="UTC")
            minutes  = {"PT60M": 60, "PT30M": 30, "PT15M": 15}.get(res_el.text, 60)
            for point in period.findall("{*}Point"):
                pos_el = point.find("{*}position")
                qty_el = point.find("{*}quantity")
                if pos_el is None or qty_el is None:
                    continue
                ts_val = start_ts + pd.Timedelta(minutes=minutes * (int(pos_el.text) - 1))
                records.append({"ts": ts_val, "qty": float(qty_el.text)})
    if not records:
        return pd.Series(dtype=float)
    return pd.DataFrame(records).set_index("ts")["qty"].sort_index()


def get_rte_french_nuclear(start=None, end=None):
    """
    Fetch French nuclear actual generation from ENTSO-E (A75, psrType B14).
    Fetches the last 3 days of actuals (R-0 from ENTSO-E), resamples to 30-min,
    and forward-fills across the forecast window.
    Returns a Series indexed by UTC timestamps named 'fr_nuclear'.

    Replaces the previous RTE eco2mix source, which has a ~5-month publication
    delay and was returning January 2026 values in June 2026.
    """
    from django.conf import settings
    token = getattr(settings, "ENTSOE_API_KEY", "")
    if not token:
        return pd.Series(dtype=float, name="fr_nuclear")

    try:
        now = pd.Timestamp.now(tz="UTC")
        p_start = (now - pd.Timedelta("3d")).normalize()
        p_end   = now.normalize() + pd.Timedelta("1d")
        resp = requests.get("https://web-api.tp.entsoe.eu/api", params={
            "securityToken": token,
            "documentType":  "A75",
            "processType":   "A16",
            "in_Domain":     "10YFR-RTE------C",
            "psrType":       "B14",
            "periodStart":   p_start.strftime("%Y%m%d%H%M"),
            "periodEnd":     p_end.strftime("%Y%m%d%H%M"),
        }, timeout=30)
        resp.raise_for_status()
        s = _parse_entsoe_qty(resp.text)
        if s.empty:
            return pd.Series(dtype=float, name="fr_nuclear")
        s = s.resample("30min").mean()
    except Exception:
        logger.exception("ENTSO-E A75: failed to fetch FR nuclear generation")
        return pd.Series(dtype=float, name="fr_nuclear")

    # Forward-fill across the forecast window (up to end)
    if start is not None and end is not None:
        idx = pd.date_range(
            start=pd.Timestamp(start).tz_convert("UTC"),
            end=pd.Timestamp(end).tz_convert("UTC"),
            freq="30min",
        )
        s = s.reindex(idx.union(s.index)).ffill().reindex(idx)
    s.name = "fr_nuclear"
    return s


def get_neso_opmr(start=None, end=None):
    """
    Fetch NESO OPMR component fields for each target date.

    Returns a DataFrame indexed by UTC-normalised date with columns:
      gen_availability, max_ic_import, opmr_total, constrained_plant

    Callers compute the per-slot surplus themselves as:
      gen_availability + max_ic_import - slot_demand - opmr_total - constrained_plant
    This avoids the old bias of anchoring to peak demand for every half-hourly slot.
    """
    try:
        url = "https://api.neso.energy/api/3/action/datastore_search"
        params = {
            "resource_id": "0eede912-8820-4c66-a58a-f7436d36b95f",
            "limit": 100,
            "sort": "_id desc",
        }
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        records = response.json().get("result", {}).get("records", [])
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        df["Date"]         = pd.to_datetime(df["Date"], utc=True)
        df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True)
        for col in ["Generator Availability", "Maximum I/C Import", "OPMR total", "Constrained Plant"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Constrained Plant"] = df["Constrained Plant"].fillna(0)
        df = df.sort_values("Publish Date").drop_duplicates(subset="Date", keep="last")
        df = df.set_index("Date").sort_index()
        today = pd.Timestamp.now(tz="UTC").normalize()
        df = df[df.index >= today - pd.Timedelta("1d")]
    except Exception:
        logger.exception("Unable to download NESO OPMR data")
        return pd.DataFrame()

    if len(df) == 0:
        return pd.DataFrame()

    out = pd.DataFrame({
        "gen_availability":   df["Generator Availability"],
        "max_ic_import":      df["Maximum I/C Import"],
        "opmr_total":         df["OPMR total"],
        "constrained_plant":  df["Constrained Plant"],
        "national_surplus":   df["National Surplus"],
    })
    out.index = out.index.normalize()
    return out


def get_melngc_margin():
    """
    Fetch BMRS Indicated Day-Ahead margin (boundary N) at settlement-period resolution.

    Returns a Series of indicatedMargin (MW) indexed by UTC startTime for ~30 hours.
    This uses actual dispatch forecasts rather than available capacity, so it's on a
    larger absolute scale than OPMR-derived surplus — stored as a SEPARATE feature.
    """
    try:
        resp = requests.get(
            "https://data.elexon.co.uk/bmrs/api/v1/forecast/indicated/day-ahead",
            params={"format": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        records = [r for r in resp.json().get("data", []) if r.get("boundary") == "N"]
        if not records:
            return pd.Series(dtype=float, name="melngc_margin")
        df = pd.DataFrame(records)
        df.index = pd.to_datetime(df["startTime"], utc=True)
        return df["indicatedMargin"].rename("melngc_margin").sort_index()
    except Exception:
        logger.exception("Unable to download BMRS MELNGC data")
        return pd.Series(dtype=float, name="melngc_margin")


def get_latest_nuclear_forecast(start=None, end=None):
    url = "https://data.elexon.co.uk/bmrs/api/v1/forecast/availability/daily"
    params = {"fuelType": "NUCLEAR", "format": "json"}
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        df = pd.DataFrame(response.json()["data"])
    except Exception:
        logger.exception("Unable to download latest nuclear availability forecast")
        return pd.Series(dtype=float, name="nuclear")

    return nuclear_availability_to_half_hourly(df, start=start, end=end)


def get_nuclear_estimate_for_forecast_date(forecast_date, created_at):
    forecast_date = pd.Timestamp(forecast_date).tz_convert("GB").normalize()
    created_at = pd.Timestamp(created_at).tz_convert("UTC")
    data = get_nuclear_availability_evolution(forecast_date)
    if len(data) == 0:
        return None

    known = data[data["publishTime"] <= created_at].sort_values("publishTime")
    if len(known) == 0:
        known = data.sort_values("publishTime").head(1)
    return float(known.iloc[-1]["outputUsable"])


def get_nuclear_availability_evolution(forecast_date):
    forecast_date = pd.Timestamp(forecast_date).tz_convert("GB").normalize()
    url = "https://data.elexon.co.uk/bmrs/api/v1/forecast/availability/daily/evolution"
    params = {
        "fuelType": "NUCLEAR",
        "forecastDate": forecast_date.strftime("%Y-%m-%d"),
        "format": "json",
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = pd.DataFrame(response.json()["data"])
    except Exception:
        logger.warning("Unable to download nuclear availability evolution for %s", forecast_date.date())
        return pd.DataFrame()

    if len(data) == 0:
        return pd.DataFrame()

    data["publishTime"] = pd.to_datetime(data["publishTime"], utc=True)
    return data


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
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
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
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
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
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
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
            "url": "https://api.neso.energy/api/3/action/datastore_search_sql",
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

    gas = get_gas_ttf_history(start=start)
    if len(gas) > 0:
        hist["gas_ttf"] = gas.reindex(hist.index, method="ffill").bfill()
    else:
        hist["gas_ttf"] = 0

    nuclear = get_latest_nuclear_forecast(start=hist.index.min(), end=hist.index.max())
    if len(nuclear) > 0:
        hist["nuclear"] = nuclear.reindex(hist.index, method="ffill").bfill()
    else:
        hist["nuclear"] = 0

    all_cols = ["total_wind", "bm_wind", "solar", "nuclear", "gas_ttf", "demand"] + meteo_cols
    missing_cols = [c for c in all_cols if c not in hist.columns]
    if len(missing_cols) > 0:
        logger.error(f">>> ERROR: No historic data for {missing_cols} ")
        return pd.DataFrame(), missing_cols
    else:
        return hist.astype(float).dropna(), missing_cols


def _neso_csv_fallback(resource_id, date_col, cols, rename, tz="UTC"):
    """
    When the NESO CKAN datastore API returns empty, fetch the resource's CSV file
    directly. Uses resource_show to get the canonical download URL so we don't
    hard-code a path that can change.
    """
    from io import StringIO

    try:
        meta = requests.get(
            "https://api.neso.energy/api/3/action/resource_show",
            params={"id": resource_id},
            timeout=30,
        )
        meta.raise_for_status()
        csv_url = meta.json()["result"]["url"]
    except Exception as exc:
        logger.warning("NESO CSV fallback: resource_show failed id=%s error=%s", resource_id, exc)
        return pd.DataFrame(), str(exc)[:80]

    try:
        r = requests.get(csv_url, timeout=120)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception as exc:
        logger.warning("NESO CSV fallback: download failed url=%s error=%s", csv_url, exc)
        return pd.DataFrame(), str(exc)[:80]

    try:
        df.index = pd.to_datetime(df[date_col])
        if df.index.tzinfo is None:
            df.index = df.index.tz_localize(tz, ambiguous="infer")
        # Drop historical data we don't need — keep last 1 day + all future
        cutoff = pd.Timestamp.now(tz=tz) - pd.Timedelta("1D")
        df = df[df.index >= cutoff]
        if cols:
            df = df[cols]
        if rename:
            df = df.set_axis(rename, axis=1)
        df = df.sort_index()
        df = df[~df.index.duplicated()]
        logger.info("NESO CSV fallback succeeded id=%s rows=%s", resource_id, len(df))
        return df, None
    except Exception as exc:
        logger.warning("NESO CSV fallback: parse failed id=%s error=%s", resource_id, exc)
        return pd.DataFrame(), str(exc)[:80]


def get_latest_forecast():
    ndf_from = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
    ndf_to = (pd.Timestamp.now().normalize() + pd.Timedelta("24h")).strftime("%Y-%m-%d")

    forecast_data = [
        {
            "api_group": "neso_wind",
            "label": "WINDFOR",
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=93c3048e-1dab-4057-a2a9-417540583929&limit=1000",
            "record_path": ["result", "records"],
            "tz": "UTC",
            "date_col": "Datetime",
            "cols": ["Wind_Forecast"],
            "rename": ["bm_wind"],
            "csv_fallback": "93c3048e-1dab-4057-a2a9-417540583929",
        },
        {
            "api_group": "neso_da_wind",
            "label": "WINDFOR-DA",
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=b2f03146-f05d-4824-a663-3a4f36090c71&limit=1000",
            "record_path": ["result", "records"],
            "tz": "UTC",
            "date_col": "Datetime_GMT",
            "cols": ["Incentive_forecast"],
            "rename": ["da_wind"],
            "csv_fallback": "b2f03146-f05d-4824-a663-3a4f36090c71",
        },
        {
            "api_group": "neso_solar",
            "label": "EMBSOLARFOR",
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=db6c038f-98af-4570-ab60-24d71ebd0ae5&limit=1000",
            "record_path": ["result", "records"],
            "tz": "UTC",
            "cols": ["EMBEDDED_SOLAR_FORECAST", "EMBEDDED_WIND_FORECAST"],
            "rename": ["solar", "emb_wind"],
            "date_col": "DATE_GMT",
            "time_col": "TIME_GMT",
            "csv_fallback": "db6c038f-98af-4570-ab60-24d71ebd0ae5",
        },
        {
            "api_group": "neso_demand",
            "label": "NATDEMAND",
            "url": "https://api.neso.energy/api/3/action/datastore_search?resource_id=7c0411cd-2714-4bb5-a408-adb065edf34d&limit=5000",
            "record_path": ["result", "records"],
            "date_col": "GDATETIME",
            "tz": "UTC",
            "cols": ["NATIONALDEMAND"],
            "csv_fallback": "7c0411cd-2714-4bb5-a408-adb065edf34d",
        },
        {
            "api_group": "openmeteo",
            "label": "Open-Meteo",
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
            "api_group": "bmrs",
            "label": "NDF",
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
    source_rows = {}
    source_details = {}  # api_group → {label, rows, error}

    def _err_str(e):
        if isinstance(e, int):
            return f"HTTP {e}"
        if e is not None:
            return str(e)[:80]
        return "no data returned"

    for x in forecast_data:
        group = x.get("api_group", "other")
        label = x.get("label", group)
        csv_fallback_id = x.get("csv_fallback")
        ds_kwargs = {k: v for k, v in x.items() if k not in ("api_group", "label", "csv_fallback")}
        data, e = DataSet(**ds_kwargs).download()
        n = len(data)

        # If the datastore API returned nothing, try CSV fallback
        used_csv_fallback = False
        if n == 0 and csv_fallback_id:
            logger.info("API returned no data for %s — trying CSV fallback", label)
            data, e = _neso_csv_fallback(
                csv_fallback_id,
                ds_kwargs.get("date_col"),
                ds_kwargs.get("cols"),
                ds_kwargs.get("rename"),
                ds_kwargs.get("tz", "UTC"),
            )
            n = len(data)
            if n > 0:
                used_csv_fallback = True

        prev = source_details.get(group, {"label": label, "rows": 0, "error": None, "fallback": False})
        prev["rows"] += n
        if n > 0:
            downloaded_data += [data]
            source_rows[group] = source_rows.get(group, 0) + n
            if used_csv_fallback:
                prev["fallback"] = True
        else:
            download_errors += [e]
            if prev["error"] is None:
                prev["error"] = _err_str(e)
        source_details[group] = prev

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

    df.loc[df["da_wind"] > 0, "bm_wind"] = df["da_wind"]
    df.drop("da_wind", axis=1, inplace=True)

    nuclear = get_latest_nuclear_forecast(start=df.index.min(), end=df.index.max())
    if len(nuclear) > 0:
        df["nuclear"] = nuclear.reindex(df.index, method="ffill").bfill()
    else:
        df["nuclear"] = 0

    gas_av = get_gas_availability_forecast(start=df.index.min(), end=df.index.max())
    if len(gas_av) > 0:
        df["gas_availability"] = gas_av.reindex(df.index, method="ffill").bfill()
    else:
        df["gas_availability"] = None

    df["gas_ttf"] = gas_ttf_at(pd.Timestamp.now(tz="UTC"))

    # French nuclear actual generation (ENTSO-E A75) — optional, nullable
    fr_nuc = get_rte_french_nuclear(start=df.index.min(), end=df.index.max())
    if len(fr_nuc) > 0:
        df["fr_nuclear"] = fr_nuc.reindex(df.index, method="ffill")
        source_rows["rte_nuclear"] = int(fr_nuc.notna().sum())
        source_details["rte_nuclear"] = {"label": "ENTSO-E FR nuclear", "rows": source_rows["rte_nuclear"], "error": None, "fallback": False}
    else:
        df["fr_nuclear"] = None
        source_rows["rte_nuclear"] = 0
        source_details["rte_nuclear"] = {"label": "ENTSO-E FR nuclear", "rows": 0, "error": "no data", "fallback": False}

    # NESO OPMR — per-slot surplus: gen_availability + max_ic - slot_demand - opmr_total
    # Uses each slot's own demand forecast rather than the daily peak demand, which removes
    # the systematic bias that arose from applying a peak-anchored value to overnight slots.
    opmr_daily = get_neso_opmr()
    if not opmr_daily.empty and "demand" in df.columns:
        # Normalise slot index to UTC midnight to match the OPMR daily index.
        # df.index may be GB-timezone (BST = UTC+1); normalize() without tz conversion
        # would give midnight GB which is off by one hour from midnight UTC.
        slot_dates = df.index.tz_convert("UTC").normalize()
        def _align(col):
            s = opmr_daily[col]
            s = s[~s.index.duplicated(keep="last")]
            return s.reindex(slot_dates).values

        df["dispatchable_capacity"] = (
            _align("gen_availability")
            + _align("max_ic_import")
            - _align("opmr_total")
            - _align("constrained_plant")
        )
        df["opmr_national_surplus"] = _align("national_surplus")
        source_rows["neso_opmr"]   = int(df["dispatchable_capacity"].notna().sum())
        source_details["neso_opmr"] = {"label": "NESO OPMR", "rows": source_rows["neso_opmr"], "error": None, "fallback": False}
    else:
        df["dispatchable_capacity"] = None
        df["opmr_national_surplus"] = None
        source_rows["neso_opmr"]   = 0
        source_details["neso_opmr"] = {"label": "NESO OPMR", "rows": 0, "error": "no data", "fallback": False}

    # MELNGC — BMRS indicated day-ahead margin at settlement-period resolution (~30 h).
    # Separate feature from dispatchable_capacity: uses actual dispatch forecasts so it's on a
    # different (larger) absolute scale; stored as melngc_margin and tested via experiment.
    melngc = get_melngc_margin()
    if not melngc.empty:
        df["melngc_margin"] = melngc.reindex(df.index)
        source_rows["melngc"]   = int(df["melngc_margin"].notna().sum())
        source_details["melngc"] = {"label": "BMRS MELNGC", "rows": source_rows["melngc"], "error": None, "fallback": False}
    else:
        df["melngc_margin"] = None
        source_rows["melngc"]   = 0
        source_details["melngc"] = {"label": "BMRS MELNGC", "rows": 0, "error": "no data", "fallback": False}

    # Open-Meteo France weather (wind+rad) — continental supply proxy, 16-day forecast
    fr_wx = get_open_meteo_fr_weather(start=df.index.min(), end=df.index.max())
    if not fr_wx.empty and fr_wx["fr_wind"].notna().any():
        df["fr_wind"] = fr_wx["fr_wind"].reindex(df.index, method="nearest")
        df["fr_rad"]  = fr_wx["fr_rad"].reindex(df.index, method="nearest")
        _fr_rows = int(fr_wx["fr_wind"].notna().sum())
        source_rows["openmeteo_fr"] = _fr_rows
        source_details["openmeteo_fr"] = {"label": "Open-Meteo FR", "rows": _fr_rows, "error": None, "fallback": False}
    else:
        df["fr_wind"] = None
        df["fr_rad"]  = None
        source_rows["openmeteo_fr"] = 0
        source_details["openmeteo_fr"] = {"label": "Open-Meteo FR", "rows": 0, "error": "no data", "fallback": False}

    all_cols = ["emb_wind", "bm_wind", "solar", "nuclear", "gas_ttf", "demand", "temp_2m", "wind_10m", "rad"]
    missing_cols += [c for c in all_cols if c not in df.columns]
    if len(missing_cols) > 0:
        logger.error("No forecast data for columns=%s", missing_cols)
        return pd.DataFrame(), missing_cols, source_rows, source_details
    else:
        df["date_time"] = pd.to_datetime(df.index)
        df["time"] = df["date_time"].dt.hour + df["date_time"].dt.minute / 60
        df["day_of_week"] = df["date_time"].dt.day_of_week.astype(int)
        # df["day_of_year"] = df["date_time"].dt.day_of_year.astype(int)

        df.index = pd.to_datetime(df.index).tz_convert("GB")
        df.drop(["date_time"], axis=1, inplace=True)

        # dropna only on required columns; fr_nuclear and dispatchable_capacity are optional
        return df.sort_index().dropna(subset=all_cols), missing_cols, source_rows, source_details


def get_weather_ensemble(n_members=10, forecast_days=3):
    """
    Fetch ICON seamless ensemble weather from Open-Meteo.
    Returns a list of DataFrames (30-min, GB timezone, columns: temp_2m, wind_10m, rad).
    Returns empty list on any failure so callers can degrade gracefully.
    """
    url = "https://ensemble-api.open-meteo.com/v1/ensemble"
    params = {
        "latitude": 54.0,
        "longitude": 2.3,
        "hourly": ["temperature_2m", "wind_speed_10m", "direct_radiation"],
        "models": "icon_seamless",
        "forecast_days": forecast_days,
    }
    try:
        response = requests.get(url=url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        logger.warning("Ensemble weather fetch failed: %s", e)
        return []

    hourly = data.get("hourly", {})
    raw_times = hourly.get("time", [])
    if not raw_times:
        logger.warning("Ensemble weather: empty response from %s", url)
        return []

    times = pd.to_datetime(raw_times).tz_localize("UTC")

    member_nums = sorted({
        int(m.group(1))
        for key in hourly
        for m in [re.match(r".+_member(\d+)$", key)]
        if m
    })
    if not member_nums:
        logger.warning("Ensemble weather: no member columns found")
        return []

    use_members = member_nums[:n_members]
    logger.info("Ensemble weather: %d members available, using %d", len(member_nums), len(use_members))

    members = []
    for num in use_members:
        suffix = f"_member{num:02d}"
        try:
            df = pd.DataFrame(
                {
                    "temp_2m": hourly.get(f"temperature_2m{suffix}", [np.nan] * len(times)),
                    "wind_10m": hourly.get(f"wind_speed_10m{suffix}", [np.nan] * len(times)),
                    "rad": hourly.get(f"direct_radiation{suffix}", [np.nan] * len(times)),
                },
                index=times,
            )
            df = df.resample("30min").interpolate(method="time")
            df.index = df.index.tz_convert("GB")
            members.append(df)
        except Exception as e:
            logger.warning("Ensemble weather: failed to process member %02d: %s", num, e)

    return members


class DataSet:
    def __init__(self, *args, **kwargs) -> None:
        self.params = kwargs.pop("params", {})
        self.tz = kwargs.pop("tz", "UTC")
        self.__dict__ = self.__dict__ | kwargs

        # self.__dict__ = self.__dict__ | kwargs

    def update(self, download_all=False, hdf=None):
        pass

    def download(self, tz="GB", params={}):
        logger.debug("Downloading dataset url=%s", self.url)
        response = None
        code = None
        for n in range(RETRIES):
            try:
                response = requests.get(url=self.url, params=self.params)
                response.raise_for_status()
                break

            except HTTPError as exc:
                code = exc.response.status_code

                if code in RETRY_CODES:
                    # retry after n seconds
                    logger.warning("Retrying dataset download url=%s status=%s attempt=%s", self.url, code, n + 1)
                    time.sleep(n)
                    continue
                logger.exception("Dataset download failed url=%s status=%s", self.url, code)
                return pd.DataFrame(), code

            except requests.exceptions.RequestException as exc:
                logger.warning("Dataset download request error url=%s attempt=%s error=%s", self.url, n + 1, exc)
                time.sleep(n)

        if response is None:
            logger.error("Dataset download failed after retries url=%s", self.url)
            return pd.DataFrame(), code

        try:
            df = pd.json_normalize(response.json(), self.record_path)
        except:
            try:
                df = pd.DataFrame(response.json()[self.record_path[0]])
            except Exception as e:
                logger.exception("Unable to parse dataset response url=%s params=%s", self.url, self.params)
                return pd.DataFrame(), code

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i = 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df.index = pd.to_datetime(df[self.date_col])
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize(self.tz, ambiguous="infer")
        except Exception as e:
            logger.exception("Unable to parse dataset datetime url=%s index=%s", self.url, df.index)

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df.index = pd.to_datetime(df["Date"]) + (df["Settlement_period"] - 1) * pd.Timedelta("30min")
            df.index = df.index.tz_localize("UTC")
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df.index += pd.to_datetime(df[self.time_col].str[:5], format="%H:%M") - pd.Timestamp("1900-01-01")
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df.index += (df[self.period_col] - 1) * pd.Timedelta("30min")
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df.index = df.index.tz_convert(tz)
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df = df[self.cols]
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            if "func" in self.__dict__:
                df = df.resample(self.resample).aggregate(self.func)
            elif "resample" in self.__dict__:
                df = df.resample(self.resample).mean()
        except Exception as e:
            logger.exception("Unable to resample dataset url=%s", self.url)

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df = df.interpolate()
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        try:
            df = df.sort_values(self.sort_col)
        except:
            pass

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

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

        if "EMBEDDED_SOLAR_FORECAST" in self.cols:
            i += 1
            logger.debug("%s embedded solar forecast rows=%s head=%s", i, len(df), df.iloc[:30].to_dict())

        df = df.sort_index()
        df = df[~df.index.duplicated()]
        return df, None


def get_agile(start=pd.Timestamp("2023-07-01"), tz="GB", region="G"):
    start = pd.Timestamp(start).tz_convert("UTC")
    product = "AGILE-24-10-01"
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


def day_ahead_to_agile(df, reverse=False, region="G", export=False):
    df.index = df.index.tz_convert("GB")
    x = pd.DataFrame(df).set_axis(["In"], axis=1)
    x["In"] = x["In"].astype(float)
    x["Out"] = x["In"]
    x["Peak"] = (x.index.hour >= 16) & (x.index.hour < 19)

    if regions.get(region, {}).get("raw_day_ahead"):
        if export:
            raise ValueError("Export pricing is not available for raw day-ahead prices")
        return x["Out"].rename("day_ahead")

    if export:
        factor, base_adder, peak_adder = regions[region]["export_factors"]
        if reverse:
            x.loc[x["Peak"], "Out"] -= peak_adder
            x["Out"] -= base_adder
            x["Out"] /= factor
        else:
            x["Out"] *= factor
            x["Out"] += base_adder
            x.loc[x["Peak"], "Out"] += peak_adder
            x["Out"] = x["Out"].clip(lower=0)

        name = "day_ahead" if reverse else "agile_export"
        return x["Out"].rename(name)

    shifts = pd.Series(GLOBAL_SETTINGS["SHIFTS"])
    shifts.index = pd.to_datetime(shifts.index).tz_localize("GB")

    unique_index = pd.DatetimeIndex(x.index.unique()).sort_values()
    shifts = pd.concat([shifts, pd.Series(index=[unique_index[-1]], data=[shifts.iloc[-1]])]).sort_index()
    shifts = shifts.resample("30min").ffill()
    shifts = shifts.reindex(shifts.index.union(unique_index)).sort_index().ffill().bfill().reindex(unique_index)
    x["Shift"] = shifts.reindex(x.index).to_numpy()

    if reverse:
        x.loc[x["Peak"], "Out"] -= regions[region]["factors"][1]
        x.loc[x["Peak"], "Out"] -= x.loc[x["Peak"], "Shift"]
        x["Out"] /= regions[region]["factors"][0]
    else:
        x["Out"] *= regions[region]["factors"][0]
        x.loc[x["Peak"], "Out"] += regions[region]["factors"][1]
        x.loc[x["Peak"], "Out"] += x.loc[x["Peak"], "Shift"]

    name = "day_ahead" if reverse else "agile"
    return x["Out"].rename(name)


def import_agile_to_export_agile(df, region="G"):
    day_ahead = day_ahead_to_agile(df, reverse=True, region=region)
    return day_ahead_to_agile(day_ahead, region=region, export=True)


def df_to_Model(df, myModel, update=False):
    # df = df.dropna()
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
            except Exception as e:
                logger.warning("Failed to update model=%s datetime=%s error=%s", myModel.__name__, index, e)


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
