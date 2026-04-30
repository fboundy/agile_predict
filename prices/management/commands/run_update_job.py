import json
import os
import traceback

from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import close_old_connections
from django.utils import timezone

from config.settings import BASE_DIR


STATUS_PATH = BASE_DIR / "logs" / "update_status.json"


def write_status(status):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    status["updated_at"] = timezone.now().isoformat()
    STATUS_PATH.write_text(json.dumps(status, indent=2, sort_keys=True))


class Command(BaseCommand):
    help = "Run the update command and write a status file for HTTP-triggered jobs."

    def add_arguments(self, parser):
        parser.add_argument("--debug", action="store_true")
        parser.add_argument("--min_fd")
        parser.add_argument("--min_ad")
        parser.add_argument("--max_days")
        parser.add_argument("--no_day_of_week", action="store_true")
        parser.add_argument("--train_frac")
        parser.add_argument("--drop_last")
        parser.add_argument("--ignore_forecast", action="append")
        parser.add_argument("--no_ranges", action="store_true")
        parser.add_argument("--skip_kde_plot", action="store_true")

    def handle(self, *args, **options):
        started_at = timezone.now().isoformat()
        update_keys = {
            "debug",
            "min_fd",
            "min_ad",
            "max_days",
            "no_day_of_week",
            "train_frac",
            "drop_last",
            "ignore_forecast",
            "no_ranges",
            "skip_kde_plot",
        }

        update_options = {}
        for key, value in options.items():
            if key not in update_keys or value in {None, ""}:
                continue
            if isinstance(value, list) and not value:
                continue
            update_options[key] = value

        write_status(
            {
                "status": "running",
                "pid": os.getpid(),
                "started_at": started_at,
                "log": str(BASE_DIR / "logs" / "update_http.log"),
            }
        )

        try:
            close_old_connections()
            call_command("update", **update_options)
        except Exception as exc:
            close_old_connections()
            write_status(
                {
                    "status": "failed",
                    "pid": os.getpid(),
                    "started_at": started_at,
                    "finished_at": timezone.now().isoformat(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "log": str(BASE_DIR / "logs" / "update_http.log"),
                }
            )
            raise

        close_old_connections()
        write_status(
            {
                "status": "completed",
                "pid": os.getpid(),
                "started_at": started_at,
                "finished_at": timezone.now().isoformat(),
                "log": str(BASE_DIR / "logs" / "update_http.log"),
            }
        )
