from django.conf import settings
from django.core.management.base import BaseCommand

from prices.timestral import run_and_store


class Command(BaseCommand):
    help = "Run the Timestral zero-shot forecast and store it as the 'timestral' external forecast (region G)."

    def add_arguments(self, parser):
        parser.add_argument("--model", default=settings.TIMESTRAL_MODEL, help="Hugging Face repo id or local path")
        parser.add_argument("--context-days", type=int, default=settings.TIMESTRAL_CONTEXT_DAYS)

    def handle(self, *args, **options):
        count = run_and_store(options["model"], options["context_days"])
        self.stdout.write(f"Saved {count} Timestral rows")
