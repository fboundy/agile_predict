from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData, AgileData

from config.utils import *


class Command(BaseCommand):
    def add_arguments(self, parser):
        # Positional arguments
        parser.add_argument("id", nargs="+", type=int)

        # Named (optional) arguments

    def handle(self, *args, **options):
        # print(args)
        # print(options)
        for id in options["id"]:
            f = Forecasts.objects.filter(id=id)
            f.using("default").delete()
