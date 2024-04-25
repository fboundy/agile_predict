from django.core.management.base import BaseCommand
from ...models import History, PriceHistory, Forecasts, ForecastData


class Command(BaseCommand):
    # def add_arguments(self, parser):
    #     parser.add_argument("poll_ids", nargs="+", type=int)

    def handle(self, *args, **options):
        PriceHistory.objects.all().delete()
        History.objects.all().delete()
        Forecasts.objects.all().delete()
        ForecastData.objects.all().delete()
