from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *
import pandas as pd
from time import sleep
import os
import logging


class Command(BaseCommand):
    def add_arguments(self, parser):
        # Positional arguments
        # parser.add_argument("poll_ids", nargs="+", type=int)

        # Named (optional) arguments
        parser.add_argument(
            "--count",
        )

    def handle(self, *args, **options):
        count = int(options.get("count", 0) or 0)

        local = "temp"
        hdf = "forecast.hdf"

        local_dir = os.path.join(os.getcwd(), local)
        hdf_path = os.path.join(local_dir, hdf)

        # Setup logging to both file and console
        log_file = os.path.join(local_dir, "import_forecast.log")
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers if they haven't been added already
        if not logger.handlers:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        if os.path.exists(hdf_path):
            ff = pd.read_hdf(hdf_path, key="Forecasts").set_index("name").sort_index()
            fd = pd.read_hdf(hdf_path, key="ForecastData")
            ph = pd.read_hdf(hdf_path, key="PriceHistory").set_index("date_time").sort_index()[["day_ahead", "agile"]]
            ad = pd.read_hdf(hdf_path, key="AgileData")

            # logger.info("Price History:")
            model_ph = [x.date_time for x in PriceHistory.objects.all()]
            # logger.info(f"{'| '.join([x.strftime('%Y-%d-%m %H:%M') for x in ph.index])}")

            # logger.info("Model Price History:")
            # logger.info(f"{'| '.join([x.strftime('%Y-%d-%m %H:%M') for x in model_ph])}")

            ph = ph.drop([i for i in model_ph if i in ph.index])

            for index, row in ph.iterrows():
                try:
                    ph_obj = PriceHistory.objects.get(date_time=index)
                except PriceHistory.DoesNotExist:
                    new_values = {"date_time": index}
                    new_values.update(row)
                    ph_obj = PriceHistory(**new_values)
                    ph_obj.save()
                    logger.info(f"{index} Added")
            logger.info(f"File price history:  {len(ph)}")
            logger.info(f"Model price history: {len(model_ph)}")

            model_ff = [x.name for x in Forecasts.objects.all()]

            logger.info(f"File forecasts:      {len(ff)}")
            logger.info(f"Model forecasts:     {len(model_ff)}")

            model_fd_count = ForecastData.objects.all().count()
            logger.info(f"File forecast data:  {len(fd)}")
            logger.info(f"Model forecast data: {model_fd_count}")

            model_ad_count = AgileData.objects.all().count()
            logger.info(f"File agile data:     {len(ad)}")
            logger.info(f"Model agile:         {model_ad_count}")

            i = 0
            if count > 0:
                logger.info(f"Processing last {count} rows of file forecasts")
                ff = ff.iloc[-count:]

            for index, row in ff.iterrows():
                i += 1
                xrow = row[["created_at"]]
                try:
                    ff_obj = Forecasts.objects.get(name=index)
                except Forecasts.DoesNotExist:
                    new_values = {"name": index}
                    new_values.update(xrow)
                    ff_obj = Forecasts(**new_values)
                    ff_obj.save()

                id = row["id"]
                name = row["name"]
                df = fd[fd["forecast_id"] == id].set_index("date_time")
                z = "exists"
                for index, row in df.iterrows():
                    yrow = row.drop(["id", "forecast_id"])
                    try:
                        new_values = {"forecast": ff_obj, "date_time": index}
                        fd_obj = ForecastData.objects.get(forecast=ff_obj, date_time=index)
                    except ForecastData.DoesNotExist:
                        new_values.update(yrow)
                        fd_obj = ForecastData(**new_values)
                        fd_obj.save()
                        z = "added"

                str_log = f"{i:4d} {id} {name} Forecast Data {z:8s} | "

                df = ad[ad["forecast_id"] == id].set_index("date_time")
                if len(df) > 0:
                    z = "exists in model."
                    for index, row in df.iterrows():
                        yrow = row.drop(["id", "forecast_id"])
                        for region in ["G", "X"]:
                            try:
                                ad_obj = AgileData.objects.get(forecast=ff_obj, date_time=index, region=region)
                            except AgileData.DoesNotExist:
                                try:
                                    new_values = {"forecast": ff_obj, "date_time": index, "region": region}
                                    new_values.update(yrow)
                                    ad_obj = AgileData(**new_values)
                                    ad_obj.save()
                                    z = "added."
                                except Exception as e:
                                    z = f"add failed: {str(e)}"
                else:
                    z = "not available in hdf."

                str_log += f" Agile Data {z}"
                logger.info(str_log)
        else:
            logger.error("Unable to open data file")

        for f in Forecasts.objects.all().order_by("-created_at"):
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            logger.info(f"{f.id} {f.name} {q.count()} {a.count()}")
