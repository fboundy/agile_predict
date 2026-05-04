import logging

import pandas as pd
from django.core.management.base import BaseCommand

from config.utils import get_gas_ttf_history, get_latest_nuclear_forecast, get_nuclear_availability_evolution, gas_ttf_at
from prices.models import ForecastData, Forecasts, History


logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Backfill estimated nuclear availability and TTF gas price features."

    def add_arguments(self, parser):
        parser.add_argument("--limit-forecasts", type=int)
        parser.add_argument("--skip-nuclear-evolution", action="store_true")

    def handle(self, *args, **options):
        forecasts = Forecasts.objects.order_by("created_at")
        if options.get("limit_forecasts"):
            forecasts = forecasts.reverse()[: options["limit_forecasts"]]

        forecast_list = list(forecasts)
        if not forecast_list:
            self.stdout.write("No forecasts to backfill.")
            return

        start = min(forecast.created_at for forecast in forecast_list) - pd.Timedelta("7D")
        end = max(forecast.created_at for forecast in forecast_list) + pd.Timedelta("1D")
        gas_history = get_gas_ttf_history(start=start, end=end)

        latest_nuclear = get_latest_nuclear_forecast()
        nuclear_by_date = {}
        nuclear_evolution_by_date = {}
        updated = 0

        for index, forecast in enumerate(forecast_list, start=1):
            gas_ttf = gas_ttf_at(forecast.created_at, gas_history=gas_history)
            rows = list(ForecastData.objects.filter(forecast=forecast).order_by("date_time"))
            for row in rows:
                target_date = pd.Timestamp(row.date_time).tz_convert("GB").normalize()
                nuclear = None
                if not options["skip_nuclear_evolution"]:
                    if target_date not in nuclear_evolution_by_date:
                        nuclear_evolution_by_date[target_date] = get_nuclear_availability_evolution(target_date)
                    evolution = nuclear_evolution_by_date[target_date]
                    if len(evolution) > 0:
                        created_at = pd.Timestamp(forecast.created_at).tz_convert("UTC")
                        known = evolution[evolution["publishTime"] <= created_at].sort_values("publishTime")
                        if len(known) == 0:
                            known = evolution.sort_values("publishTime").head(1)
                        nuclear = float(known.iloc[-1]["outputUsable"])
                if nuclear is None:
                    nuclear = nuclear_by_date.get(target_date)
                if nuclear is None and len(latest_nuclear) > 0:
                    latest_for_date = latest_nuclear[latest_nuclear.index.normalize() == target_date]
                    if len(latest_for_date) > 0:
                        nuclear = float(latest_for_date.iloc[0])
                if nuclear is None:
                    nuclear = 0
                nuclear_by_date[target_date] = nuclear

                row.nuclear = nuclear
                row.gas_ttf = gas_ttf

            ForecastData.objects.bulk_update(rows, ["nuclear", "gas_ttf"], batch_size=1000)
            updated += len(rows)
            self.stdout.write(f"{index:4d}/{len(forecast_list)} {forecast.name}: {len(rows)} rows")

        history_rows = list(History.objects.order_by("date_time"))
        if history_rows:
            history_start = history_rows[0].date_time
            history_end = history_rows[-1].date_time
            history_gas = get_gas_ttf_history(start=history_start, end=history_end)
            history_nuclear = get_latest_nuclear_forecast(start=history_start, end=history_end)
            for row in history_rows:
                dt = pd.Timestamp(row.date_time).tz_convert("GB")
                if len(history_gas) > 0:
                    gas = history_gas[history_gas.index <= dt]
                    row.gas_ttf = float(gas.iloc[-1]) if len(gas) > 0 else float(history_gas.iloc[0])
                else:
                    row.gas_ttf = 0
                if len(history_nuclear) > 0:
                    nuclear = history_nuclear.reindex([dt], method="ffill")
                    row.nuclear = float(nuclear.iloc[0]) if len(nuclear) > 0 and pd.notna(nuclear.iloc[0]) else 0
                else:
                    row.nuclear = 0
            History.objects.bulk_update(history_rows, ["nuclear", "gas_ttf"], batch_size=1000)

        self.stdout.write(self.style.SUCCESS(f"Backfilled {updated} forecast data rows."))
