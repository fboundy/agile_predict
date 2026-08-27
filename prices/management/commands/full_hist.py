from sklearn.metrics import mean_squared_error as MSE
import numpy as np

from django.core.management.base import BaseCommand
from ...environment import require_non_production
from ...models import History

from config.utils import *

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50
MIN_HIST = 7
MAX_HIST = 28


class Command(BaseCommand):
    def handle(self, *args, **options):
        # History is dev-only data; see prices/environment.py and
        # docs/MODEL_DYNAMIC_RANGE.md. Guarded before any download.
        require_non_production("Rebuilding History")

        new_hist, _ = get_latest_history(start=pd.Timestamp("2023-07-01", tz="GB"))
        if len(new_hist) > 0:
            print(new_hist)
            History.objects.all().delete()
            df_to_Model(new_hist, History)

        else:
            print("None")
