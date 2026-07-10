import json

from django.core.management.base import BaseCommand, CommandError

from prices.models import UpdateJob


class Command(BaseCommand):
    help = (
        "Print the most recently computed feature-experiment result (winner feature set + "
        "cross-validation results) as JSON, for pasting into apply_feature_experiment on "
        "production. Run this after `manage.py update --force_experiment` locally."
    )

    def handle(self, *args, **options):
        job = (
            UpdateJob.objects.filter(job_type=UpdateJob.JOB_UPDATE)
            .exclude(options__feature_experiment=None)
            .order_by("-requested_at")
            .first()
        )
        if job is None:
            raise CommandError("No UpdateJob with a feature_experiment result found.")

        self.stdout.write(json.dumps(job.options["feature_experiment"]))
