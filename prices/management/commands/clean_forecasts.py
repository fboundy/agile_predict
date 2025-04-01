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

    def handle(self, *args, **options):
        delete = options.get("delete", False)
        forecast_days = {}
        for f in Forecasts.objects.all().order_by("-created_at"):
            q = ForecastData.objects.filter(forecast=f)
            a = AgileData.objects.filter(forecast=f)

            print(f.id, f.name, q.count(), a.count())
            if q.count() < 600 or a.count() < 8000:
                f.delete()
            else:
                if f.created_at.date() in forecast_days:
                    forecast_days[f.created_at.date()].append(f)
                else:
                    forecast_days[f.created_at.date()] = [f]

        print(forecast_days)

        keep = []
        for d in forecast_days:
            if (pd.Timestamp.now() - pd.Timestamp(d)).days <= 90:
                # print(d)
                t = [f.created_at for f in forecast_days[d]]
                id = [f.id for f in forecast_days[d]]
                df = pd.Series(index=t, data=id)
                df.index = df.index.tz_convert("GB")

                z = df[df.index.hour >= 9]
                if len(z) > 0:
                    keep.append(z.sort_index().iloc[0])
                else:
                    keep.append(df.sort_index().iloc[-1])

        print(keep)

        if delete:
            forecasts_to_delete = Forecasts.objects.exclude(id__in=keep)
            print(f"deleting ({forecasts_to_delete})")
            forecasts_to_delete.delete()
