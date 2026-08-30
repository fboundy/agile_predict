"""Rebuild `History` from NESO/Open-Meteo archives.

DEV ONLY — `History` is dev-only data, held on the CT so production's database
stays small (see docs/MODEL_DYNAMIC_RANGE.md). The guard is `.dockerignore`: this
file is excluded from the build context, so it is not present in the production
image and cannot be run there. `prices/tests.py::DockerignoreGuardTests` asserts
that entry survives.

Note this command's demand sources are stale — only the 2023 and 2024 per-year
"Historic Demand Data" resources are listed, so any window past 2024 loses demand
and the frame empties. Prefer `backfill_history`, which paginates and knows the
2025/2026 resources.
"""

from sklearn.metrics import mean_squared_error as MSE
import numpy as np

from django.core.management.base import BaseCommand
from ...models import History

from config.utils import *

DAYS_TO_INCLUDE = 7
MODEL_ITERS = 50
MIN_HIST = 7
MAX_HIST = 28


class Command(BaseCommand):
    def handle(self, *args, **options):
        new_hist, _ = get_latest_history(start=pd.Timestamp("2023-07-01", tz="GB"))
        if len(new_hist) > 0:
            print(new_hist)
            History.objects.all().delete()
            df_to_Model(new_hist, History)

        else:
            print("None")
