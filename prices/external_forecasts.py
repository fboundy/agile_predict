import logging
from datetime import timedelta, timezone as datetime_timezone

import requests
from django.db import transaction
from django.utils import timezone
from django.utils.dateparse import parse_datetime

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
    if _downloaded_today(source):
        logger.info("Skipping AgileForecast download; already downloaded today")
        return 0

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
    response = requests.get(url, timeout=15)
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
    if _downloaded_today(source):
        logger.info("Skipping X2R download; already downloaded today")
        return 0

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


def fetch_x2r(region):
    url = f"https://api.x2r.uk/agile/{region.upper()}"
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
    return {
        "source": ExternalForecast.SOURCE_X2R,
        "name": f"X2R {payload.get('region', region.upper())} {payload.get('forecast_at', '')}",
        "source_created_at": source_created_at,
        "rows": rows,
    }


def download_daily_external_forecasts():
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

    _cleanup()
    return counts
