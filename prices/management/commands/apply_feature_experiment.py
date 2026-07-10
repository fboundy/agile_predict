import json

from django.core.management.base import BaseCommand, CommandError

from prices.forecast_features import EXPERIMENT_FEATURE_SETS
from prices.models import UpdateJob


class Command(BaseCommand):
    help = (
        "Apply a feature-experiment result (JSON from export_feature_experiment, run "
        "locally) to production: writes it into the most recent UpdateJob's options so "
        "the next scheduled update picks up the winning feature set without re-running "
        "the experiment here."
    )

    def add_arguments(self, parser):
        parser.add_argument("payload", help="JSON produced by export_feature_experiment")

    def handle(self, *args, **options):
        try:
            payload = json.loads(options["payload"])
        except json.JSONDecodeError as exc:
            raise CommandError(f"Invalid JSON payload: {exc}")

        missing = {"date", "feature_set", "results"} - payload.keys()
        if missing:
            raise CommandError(f"Payload missing required keys: {sorted(missing)}")

        if payload["feature_set"] not in EXPERIMENT_FEATURE_SETS:
            raise CommandError(
                f"Unknown feature_set {payload['feature_set']!r}; "
                f"expected one of {sorted(EXPERIMENT_FEATURE_SETS)}"
            )

        job = UpdateJob.objects.filter(job_type=UpdateJob.JOB_UPDATE).order_by("-requested_at").first()
        if job is None:
            raise CommandError("No UpdateJob exists to attach the feature_experiment result to.")

        job.options["feature_experiment"] = payload
        job.save(update_fields=["options"])
        self.stdout.write(
            f"Applied feature_experiment (feature_set={payload['feature_set']!r}, "
            f"date={payload['date']}) to UpdateJob {job.id}"
        )
