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

        # for x in [hdf, hdf_path]:
        #     try:
        #         os.remove(x)
        #     except:
        #         pass

        # cmds = [
        #     'flyctl ssh console -C "python manage.py export_local"',
        #     f"flyctl sftp get {local}/{hdf}",
        #     # f"mv {hdf} {local}",
        # ]

        # for cmd in cmds:
        #     os.system(cmd)
        #     sleep(1)

        # os.rename(hdf, hdf_path)
        if os.path.exists(hdf_path):

            ff = pd.read_hdf(hdf_path, key="Forecasts").set_index("name").sort_index()
            fd = pd.read_hdf(hdf_path, key="ForecastData")
            ph = pd.read_hdf(hdf_path, key="PriceHistory").set_index("date_time").sort_index()[["day_ahead", "agile"]]
            ad = pd.read_hdf(hdf_path, key="AgileData")

            # fd = fd[fd["forecast_id"].astype(int) < ff.index.max()]

            print("Price History:")
            model_ph = [x.date_time for x in PriceHistory.objects.all()]
            print(ph.index)
            print(model_ph)
            ph = ph.drop([i for i in model_ph if i in ph.index])
            for index, row in ph.iterrows():
                try:
                    ph_obj = PriceHistory.objects.get(date_time=index)
                    # for key, value in row.items():
                    #     setattr(ph_obj, key, value)
                    # ph_obj.save()
                    # print(f"{index} Updated")
                except PriceHistory.DoesNotExist:
                    new_values = {"date_time": index}
                    new_values.update(row)
                    ph_obj = PriceHistory(**new_values)
                    ph_obj.save()
                    print(f"{index} Added")

            model_ff = [x.name for x in Forecasts.objects.all()]
            # ff = ff.drop([x for x in model_ff if x in ff.index])

            for index, row in ff.iterrows():
                xrow = row[["created_at"]]
                try:
                    ff_obj = Forecasts.objects.get(name=index)
                    # for key, value in xrow.items():
                    #     setattr(ff_obj, key, value)
                    # ff_obj.save()
                except Forecasts.DoesNotExist:
                    new_values = {"name": index}
                    new_values.update(xrow)
                    ff_obj = Forecasts(**new_values)
                    ff_obj.save()

                id = row["id"]
                df = fd[fd["forecast_id"] == id].set_index("date_time")
                z = "exists"
                for index, row in df.iterrows():
                    yrow = row.drop(["id", "forecast_id"])
                    try:
                        new_values = {"forecast": ff_obj, "date_time": index}
                        fd_obj = ForecastData.objects.get(forecast=ff_obj, date_time=index)
                        # for key, value in yrow.items():
                        #     setattr(fd_obj, key, value)
                        # fd_obj.save()

                    except ForecastData.DoesNotExist:
                        new_values.update(yrow)
                        fd_obj = ForecastData(**new_values)
                        fd_obj.save()
                        z = "added "

                str_log = f"{id} {new_values['date_time']} Forecast Data {z}"

                df = ad[ad["forecast_id"] == id].set_index("date_time")
                if len(df) > 0:
                    z = "exists"
                    for index, row in df.iterrows():
                        yrow = row.drop(["id", "forecast_id"])
                        for region in ["G", "X"]:
                            try:
                                ad_obj = AgileData.objects.get(forecast=ff_obj, date_time=index, region=region)
                                # for key, value in yrow.items():
                                #     setattr(fd_obj, key, value)
                                # fd_obj.save()

                            except AgileData.DoesNotExist:
                                new_values = {"forecast": ff_obj, "date_time": index, "region": region}
                                new_values.update(yrow)
                                ad_obj = AgileData(**new_values)
                                ad_obj.save()
                                z = "added"
                    str_log += f" Agile Data {z}"

                print(str_log)
        else:
            print("Unable to open data file")

        for f in Forecasts.objects.all().order_by("-created_at"):
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            print(f.id, f.name, q.count(), a.count())
