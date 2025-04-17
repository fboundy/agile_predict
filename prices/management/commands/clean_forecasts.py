from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *


class Command(BaseCommand):
    def add_arguments(self, parser):
        # Positional arguments
        # parser.add_argument("poll_ids", nargs="+", type=int)

        # Named (optional) arguments
        parser.add_argument(
            "--delete",
            action="store_true",
            help="Only keep one forecast per day ",
        )

        parser.add_argument(
            "--min_fd",
        )

        parser.add_argument(
            "--min_ad",
        )

        parser.add_argument(
            "--days",
        )

    def handle(self, *args, **options):
        delete = options.get("delete", False)
        min_fd = int(options.get("min_fd", 600) or 600)
        min_ad = int(options.get("min_ad", 0) or 0)
        max_days = int(options.get("days", 100000) or 10000)

        print(f"Max days: {max_days}")

        print(f"  ID  |       Name       |  #FD  |   #AD   | Days | Mean  | Stdev |")
        print(f"------+------------------+-------+---------+------+-------+-------+")
        keep = []
        for f in Forecasts.objects.all().order_by("-created_at"):
            fd = ForecastData.objects.filter(forecast=f)
            ad = AgileData.objects.filter(forecast=f)
            dt = pd.to_datetime(f.name).tz_localize("GB")
            days = (pd.Timestamp.now(tz="GB") - dt).days
            if fd.count() < min_fd or ad.count() < min_ad:
                fail = " <- Fail"
            else:
                fail = " <- Manual"
                if days < max_days:
                    for hour in [6, 10, 16, 22]:
                        if f"{hour:02d}:15" in f.name:
                            keep.append(f.id)
                            fail = ""

            if f.mean is None:
                print(f"{f.id:5d} | {f.name} | {fd.count():5d} | {ad.count():7d} | {days:4d} |  N/A  |  N/A  | {fail}")
            else:
                print(
                    f"{f.id:5d} | {f.name} | {fd.count():5d} | {ad.count():7d} | {days:4d} | {f.mean:5.2f} | {f.stdev:5.2f} | {fail}"
                )

        print(keep)

        if delete:
            forecasts_to_delete = Forecasts.objects.exclude(id__in=keep)
            print(f"deleting ({forecasts_to_delete})")
            forecasts_to_delete.delete()
