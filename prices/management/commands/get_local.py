from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *
import pandas as pd
from time import sleep
import os


class Command(BaseCommand):
    def handle(self, *args, **options):
        local = ".local"
        hdf = "forecast.hdf"

        local_dir = os.path.join(os.getcwd(), local)
        hdf_path = os.path.join(local_dir, hdf)

        for x in [hdf, hdf_path]:
            try:
                os.remove(x)
            except:
                pass

        cmds = [
            'flyctl ssh console -C "python manage.py export_local"',
            f"flyctl sftp get {local}/{hdf}",
            # f"mv {hdf} {local}",
        ]

        for cmd in cmds:
            os.system(cmd)
            sleep(1)

        os.rename(hdf, hdf_path)
