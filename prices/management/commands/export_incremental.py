import gzip
import json
from pathlib import Path

from django.core.management.base import BaseCommand
from django.forms.models import model_to_dict
from django.utils import timezone
from django.utils.dateparse import parse_datetime

from config.settings import BASE_DIR
from prices.models import AgileData, ForecastData, Forecasts, PriceHistory


DEFAULT_DIR = BASE_DIR / ".local" / "incremental"
DEFAULT_STATE = DEFAULT_DIR / "state.json"


def serialize_value(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def serialize_row(model_name, row, lookup_fields):
    fields = {key: serialize_value(value) for key, value in row.items()}
    lookup = {key: fields[key] for key in lookup_fields}
    return {"model": model_name, "lookup": lookup, "fields": fields}


def read_state(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def state_datetime(state, key):
    value = state.get(key)
    if not value:
        return None
    return parse_datetime(value)


def write_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


class Command(BaseCommand):
    help = "Export PriceHistory and new Forecast rows since the last incremental backup marker."

    def add_arguments(self, parser):
        parser.add_argument("--state", default=str(DEFAULT_STATE))
        parser.add_argument("--output")
        parser.add_argument("--no-update-state", action="store_true")
        parser.add_argument(
            "--initialize-state",
            action="store_true",
            help="Write the current high-water marks to the state file without exporting rows.",
        )

    def handle(self, *args, **options):
        state_path = Path(options["state"])
        state = read_state(state_path)
        if options["initialize_state"]:
            next_state = current_state()
            write_state(state_path, next_state)
            self.stdout.write(f"Initialized incremental backup state: {state_path}")
            self.stdout.write(json.dumps(next_state, indent=2, sort_keys=True))
            return

        output_path = Path(options["output"]) if options.get("output") else self.default_output_path()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        last_price_history = state_datetime(state, "last_price_history_date_time")
        last_forecast_created = state_datetime(state, "last_forecast_created_at")

        price_history = PriceHistory.objects.order_by("date_time")
        if last_price_history:
            price_history = price_history.filter(date_time__gte=last_price_history)
        price_history_rows = [
            row
            for row in price_history.values("date_time", "day_ahead", "agile")
            if last_price_history is None or row["date_time"] > last_price_history
        ]

        forecasts = Forecasts.objects.order_by("created_at", "id")
        if last_forecast_created:
            forecasts = forecasts.filter(created_at__gte=last_forecast_created)
        forecasts = [
            forecast
            for forecast in forecasts
            if last_forecast_created is None or forecast.created_at > last_forecast_created
        ]

        forecast_ids = [forecast.id for forecast in forecasts]
        forecast_data = ForecastData.objects.filter(forecast_id__in=forecast_ids).order_by("forecast_id", "date_time")
        agile_data = AgileData.objects.filter(forecast_id__in=forecast_ids).order_by("forecast_id", "region", "date_time")

        counts = {
            "PriceHistory": len(price_history_rows),
            "Forecasts": len(forecast_ids),
            "ForecastData": forecast_data.count(),
            "AgileData": agile_data.count(),
        }

        max_price_history = None
        max_forecast_created = None
        with gzip.open(output_path, "wt", encoding="utf-8") as handle:
            metadata = {
                "model": "__metadata__",
                "fields": {
                    "created_at": timezone.now().isoformat(),
                    "state_before": state,
                    "counts": counts,
                },
            }
            handle.write(json.dumps(metadata, sort_keys=True) + "\n")

            for row in price_history_rows:
                max_price_history = row["date_time"]
                handle.write(json.dumps(serialize_row("PriceHistory", row, ["date_time"]), sort_keys=True) + "\n")

            for forecast in forecasts:
                row = model_to_dict(forecast, fields=["name", "mean", "stdev"])
                row["created_at"] = forecast.created_at
                handle.write(json.dumps(serialize_row("Forecasts", row, ["name"]), sort_keys=True) + "\n")
                max_forecast_created = forecast.created_at

            for row in forecast_data.values(
                "forecast__name",
                "date_time",
                "day_ahead",
                "day_ahead_classified",
                "plunge_probability",
                "bm_wind",
                "solar",
                "emb_wind",
                "nuclear",
                "gas_ttf",
                "temp_2m",
                "wind_10m",
                "rad",
                "demand",
            ):
                row["forecast_name"] = row.pop("forecast__name")
                handle.write(
                    json.dumps(serialize_row("ForecastData", row, ["forecast_name", "date_time"]), sort_keys=True)
                    + "\n"
                )

            for row in agile_data.values(
                "forecast__name",
                "region",
                "date_time",
                "agile_pred",
                "agile_low",
                "agile_high",
            ):
                row["forecast_name"] = row.pop("forecast__name")
                handle.write(
                    json.dumps(
                        serialize_row("AgileData", row, ["forecast_name", "region", "date_time"]),
                        sort_keys=True,
                    )
                    + "\n"
                )

        next_state = dict(state)
        if max_price_history is not None:
            next_state["last_price_history_date_time"] = max_price_history.isoformat()
        if max_forecast_created is not None:
            next_state["last_forecast_created_at"] = max_forecast_created.isoformat()

        if not options["no_update_state"]:
            write_state(state_path, next_state)

        self.stdout.write(f"Exported incremental backup: {output_path}")
        for model_name, count in counts.items():
            self.stdout.write(f"  {model_name}: {count}")

    def default_output_path(self):
        stamp = timezone.now().strftime("%Y-%m-%d_%H-%M-%S")
        return DEFAULT_DIR / f"incremental_{stamp}.jsonl.gz"


def current_state():
    latest_price_history = PriceHistory.objects.order_by("-date_time").values_list("date_time", flat=True).first()
    latest_forecast_created = Forecasts.objects.order_by("-created_at").values_list("created_at", flat=True).first()
    state = {}
    if latest_price_history is not None:
        state["last_price_history_date_time"] = latest_price_history.isoformat()
    if latest_forecast_created is not None:
        state["last_forecast_created_at"] = latest_forecast_created.isoformat()
    return state
