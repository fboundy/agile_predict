import logging
from datetime import timedelta, timezone as datetime_timezone

import requests
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

import pandas as pd

from config.utils import day_ahead_to_agile
from prices.models import ExternalForecast


logger = logging.getLogger(__name__)

REGION = "G"
RETENTION_DAYS = 31


def _parse_timestamp(value):
    timestamp = parse_datetime(value)
    if timestamp is None:
        raise ValueError(f"Unable to parse timestamp: {value}")
    if timezone.is_naive(timestamp):
        timestamp = timezone.make_aware(timestamp, datetime_timezone.utc)
    return timestamp


def _downloaded_today(source, now=None):
    now = now or timezone.now()
    return ExternalForecast.objects.filter(
        source=source,
        region=REGION,
        downloaded_at__date=now.date(),
    ).exists()


def _cleanup(now=None):
    cutoff = (now or timezone.now()) - timedelta(days=RETENTION_DAYS)
    deleted_count, _ = ExternalForecast.objects.filter(source_created_at__lt=cutoff).delete()
    if deleted_count:
        logger.info("Deleted %s old external forecast row(s)", deleted_count)


def _save_rows(source, region, forecast_name, source_created_at, rows):
    if not rows:
        logger.warning("No %s forecast rows to save", source)
        return 0

    objects = [
        ExternalForecast(
            source=source,
            region=region,
            forecast_name=forecast_name,
            source_created_at=source_created_at,
            date_time=row["date_time"],
            agile_pred=row["agile_pred"],
            agile_low=row.get("agile_low"),
            agile_high=row.get("agile_high"),
        )
        for row in rows
    ]

    with transaction.atomic():
        ExternalForecast.objects.filter(
            source=source,
            region=region,
            source_created_at=source_created_at,
        ).delete()
        ExternalForecast.objects.bulk_create(objects, batch_size=1000)

    return len(objects)


def download_agileforecast_region_g():
    source = ExternalForecast.SOURCE_AGILEFORECAST

    url = "https://agileforecast.co.uk/api/G/"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0] if payload else {}

    source_created_at = _parse_timestamp(payload["created_at"])
    rows = [
        {
            "date_time": _parse_timestamp(row["date_time"]),
            "agile_pred": float(row["agile_pred"]),
            "agile_low": float(row["agile_low"]) if row.get("agile_low") is not None else None,
            "agile_high": float(row["agile_high"]) if row.get("agile_high") is not None else None,
        }
        for row in payload.get("prices", [])
    ]

    count = _save_rows(source, REGION, payload.get("name", ""), source_created_at, rows)
    logger.info("Downloaded AgileForecast region G rows=%s created_at=%s", count, source_created_at)
    return count


def fetch_agileforecast(region):
    url = f"https://agileforecast.co.uk/api/{region.upper()}/"
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, list):
        payload = payload[0] if payload else {}

    source_created_at = _parse_timestamp(payload["created_at"])
    rows = [
        {
            "date_time": _parse_timestamp(row["date_time"]),
            "agile_pred": float(row["agile_pred"]),
            "agile_low": float(row["agile_low"]) if row.get("agile_low") is not None else None,
            "agile_high": float(row["agile_high"]) if row.get("agile_high") is not None else None,
        }
        for row in payload.get("prices", [])
    ]
    return {
        "source": ExternalForecast.SOURCE_AGILEFORECAST,
        "name": payload.get("name", "AgileForecast"),
        "source_created_at": source_created_at,
        "rows": rows,
    }


def download_x2r_region_g():
    source = ExternalForecast.SOURCE_X2R

    url = "https://api.x2r.uk/agile/G"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()

    source_created_at = _parse_timestamp(payload["forecast_at"])
    rows = [
        {
            "date_time": _parse_timestamp(row["date"]),
            "agile_pred": float(row["price"]),
        }
        for row in payload.get("prices", {}).get("forecast", [])
    ]

    forecast_name = f"X2R {payload.get('region', REGION)} {payload.get('forecast_at', '')}"
    count = _save_rows(source, REGION, forecast_name, source_created_at, rows)
    logger.info("Downloaded X2R region G rows=%s created_at=%s", count, source_created_at)
    return count


def _convert_x2r_region_g_rows_to_national_average(rows):
    if not rows:
        return rows

    source = pd.Series(
        data=[row["agile_pred"] for row in rows],
        index=pd.to_datetime([row["date_time"] for row in rows]),
    )
    day_ahead = day_ahead_to_agile(source, reverse=True, region="G")
    national_average = day_ahead_to_agile(day_ahead, region="X")
    return [
        {
            **row,
            "agile_pred": float(national_average.loc[pd.Timestamp(row["date_time"]).tz_convert("GB")]),
        }
        for row in rows
    ]


def fetch_x2r(region):
    requested_region = region.upper()
    source_region = "G" if requested_region == "X" else requested_region
    url = f"https://api.x2r.uk/agile/{source_region}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()

    source_created_at = _parse_timestamp(payload["forecast_at"])
    rows = [
        {
            "date_time": _parse_timestamp(row["date"]),
            "agile_pred": float(row["price"]),
            "agile_low": None,
            "agile_high": None,
        }
        for row in payload.get("prices", {}).get("forecast", [])
    ]
    if requested_region == "X":
        rows = _convert_x2r_region_g_rows_to_national_average(rows)

    return {
        "source": ExternalForecast.SOURCE_X2R,
        "name": f"X2R {requested_region} {payload.get('forecast_at', '')}",
        "source_created_at": source_created_at,
        "rows": rows,
    }


def _prune_history(now=None):
    """Keep, per (source, region), only the newest source_created_at (drives the
    live comparison overlay) plus the earliest source_created_at of each day (the
    day-ahead snapshot used for accuracy history). Delete intra-day extras so the
    table stays ~1 row-set/day/source even though we refresh on every update run.
    """
    for source, region in ExternalForecast.objects.values_list("source", "region").distinct():
        created = list(
            ExternalForecast.objects.filter(source=source, region=region)
            .values_list("source_created_at", flat=True)
            .distinct()
        )
        if len(created) <= 1:
            continue
        keep = {max(created)}  # newest snapshot -> the chart
        earliest_by_day = {}
        for ts in created:
            day = timezone.localtime(ts).date()
            if day not in earliest_by_day or ts < earliest_by_day[day]:
                earliest_by_day[day] = ts
        keep.update(earliest_by_day.values())  # first-of-day -> history
        ExternalForecast.objects.filter(source=source, region=region).exclude(
            source_created_at__in=keep
        ).delete()


def refresh_external_forecasts():
    """Pull the current external forecasts and store them. Called on each update
    run (not per web request), so the comparison overlay is served from the DB
    with no live call on the request path. Retention keeps the latest snapshot
    for the chart and the first-of-day snapshot for accuracy history.
    """
    counts = {}
    for source, downloader in [
        (ExternalForecast.SOURCE_AGILEFORECAST, download_agileforecast_region_g),
        (ExternalForecast.SOURCE_X2R, download_x2r_region_g),
    ]:
        try:
            counts[source] = downloader()
        except Exception:
            logger.exception("Unable to download %s external forecast", source)
            counts[source] = 0

    _prune_history()
    _cleanup()
    return counts


# Backwards-compatible alias (older callers / cron).
download_daily_external_forecasts = refresh_external_forecasts


def region_rows_from_g(rows, region):
    """Derive a region's Agile comparison series from stored region-G rows, using
    the same national<->regional conversion the app applies to its own forecasts.
    Region G is returned unchanged; other regions (incl. X) are converted."""
    region = (region or "G").upper()
    if region == "G" or not rows:
        return rows

    idx = pd.to_datetime([row["date_time"] for row in rows])

    def _convert(key):
        raw = [row.get(key) for row in rows]
        if all(v is None for v in raw):
            return [None] * len(rows)
        series = pd.Series(
            data=[float(v) if v is not None else float("nan") for v in raw],
            index=idx,
        )
        day_ahead = day_ahead_to_agile(series, reverse=True, region="G")
        regional = day_ahead_to_agile(day_ahead, region=region)
        return [None if pd.isna(v) else float(v) for v in regional.values]

    pred = _convert("agile_pred")
    low = _convert("agile_low")
    high = _convert("agile_high")
    return [
        {**row, "agile_pred": pred[i], "agile_low": low[i], "agile_high": high[i]}
        for i, row in enumerate(rows)
    ]
