import gzip
import json
from pathlib import Path

from django.contrib.auth.models import Group, Permission, User
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_datetime

from prices.models import AgileData, ForecastData, Forecasts, PriceHistory


DATETIME_FIELDS = {
    "PriceHistory": {"date_time"},
    "Forecasts": {"created_at"},
    "ForecastData": {"date_time"},
    "AgileData": {"date_time"},
    "AuthUser": {"last_login", "date_joined"},
}


def parse_value(model_name, field, value):
    if value is None:
        return None
    if field in DATETIME_FIELDS.get(model_name, set()):
        return parse_datetime(value)
    return value


class Command(BaseCommand):
    help = "Import an incremental JSONL backup produced by export_incremental."

    def add_arguments(self, parser):
        parser.add_argument("path")

    def handle(self, *args, **options):
        path = Path(options["path"])
        if not path.exists():
            raise SystemExit(f"Incremental backup not found: {path}")

        counts = {
            "PriceHistory": 0,
            "Forecasts": 0,
            "ForecastData": 0,
            "AgileData": 0,
            "AuthUser": 0,
            "AuthGroup": 0,
            "AuthUserGroup": 0,
            "AuthGroupPermission": 0,
            "AuthUserPermission": 0,
        }
        forecast_cache = {}

        with transaction.atomic():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    model_name = item["model"]
                    if model_name == "__metadata__":
                        continue
                    fields = {
                        key: parse_value(model_name, key, value)
                        for key, value in item["fields"].items()
                    }

                    if model_name == "AuthGroup":
                        Group.objects.update_or_create(
                            name=fields.pop("name"),
                            defaults=fields,
                        )
                    elif model_name == "AuthUser":
                        User.objects.update_or_create(
                            username=fields.pop("username"),
                            defaults=fields,
                        )
                    elif model_name == "AuthUserGroup":
                        user = User.objects.get(username=fields["username"])
                        group = Group.objects.get(name=fields["group_name"])
                        user.groups.add(group)
                    elif model_name == "AuthGroupPermission":
                        group = Group.objects.get(name=fields["group_name"])
                        permission = self.get_permission(fields)
                        group.permissions.add(permission)
                    elif model_name == "AuthUserPermission":
                        user = User.objects.get(username=fields["username"])
                        permission = self.get_permission(fields)
                        user.user_permissions.add(permission)
                    elif model_name == "PriceHistory":
                        PriceHistory.objects.update_or_create(
                            date_time=fields.pop("date_time"),
                            defaults=fields,
                        )
                    elif model_name == "Forecasts":
                        Forecasts.objects.update_or_create(
                            name=fields.pop("name"),
                            defaults=fields,
                        )
                    elif model_name == "ForecastData":
                        forecast = self.get_forecast(forecast_cache, fields.pop("forecast_name"))
                        ForecastData.objects.update_or_create(
                            forecast=forecast,
                            date_time=fields.pop("date_time"),
                            defaults=fields,
                        )
                    elif model_name == "AgileData":
                        forecast = self.get_forecast(forecast_cache, fields.pop("forecast_name"))
                        AgileData.objects.update_or_create(
                            forecast=forecast,
                            region=fields.pop("region"),
                            date_time=fields.pop("date_time"),
                            defaults=fields,
                        )
                    else:
                        raise SystemExit(f"Unknown incremental backup model: {model_name}")

                    counts[model_name] += 1

        self.stdout.write(f"Imported incremental backup: {path}")
        for model_name, count in counts.items():
            self.stdout.write(f"  {model_name}: {count}")

    def get_forecast(self, cache, name):
        if name not in cache:
            cache[name] = Forecasts.objects.get(name=name)
        return cache[name]

    def get_permission(self, fields):
        return Permission.objects.get(
            content_type__app_label=fields["permission_app_label"],
            content_type__model=fields["permission_model"],
            codename=fields["permission_codename"],
        )
