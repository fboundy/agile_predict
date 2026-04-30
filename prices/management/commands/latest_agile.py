from django.core.management.base import BaseCommand
from django.db import close_old_connections

from config.utils import *


class Command(BaseCommand):
    help = "Fetch the latest Octopus Agile prices and store any new price rows."

    def handle(self, *args, **options):
        self.stdout.write("Getting historic Agile prices")
        close_old_connections()
        prices, start = model_to_df(PriceHistory)
        self.stdout.write(f"Existing price rows: {len(prices)}")
        self.stdout.write(f"Fetching Agile prices from: {start}")

        close_old_connections()
        agile = get_agile(start=start)
        self.stdout.write(f"Fetched Agile rows: {len(agile)}")

        day_ahead = day_ahead_to_agile(agile, reverse=True)

        new_prices = pd.concat([day_ahead, agile], axis=1)
        if len(prices) > 0:
            new_prices = new_prices[new_prices.index > prices.index[-1]]

        self.stdout.write(f"New price rows to write: {len(new_prices)}")
        if len(new_prices) == 0:
            self.stdout.write("No new Agile prices to write")
            return

        self.stdout.write(f"Writing Agile prices from {new_prices.index[0]} to {new_prices.index[-1]}")
        close_old_connections()
        df_to_Model(new_prices, PriceHistory)
        close_old_connections()
        self.stdout.write("Finished latest Agile price update")
