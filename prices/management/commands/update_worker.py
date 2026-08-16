from contextlib import redirect_stderr, redirect_stdout
from datetime import timedelta
import logging
import os
from pathlib import Path
import time
import traceback

from django.conf import settings
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import DatabaseError, close_old_connections, transaction
from django.utils import timezone

from config.settings import BASE_DIR
from prices.models import UpdateJob


logger = logging.getLogger("prices.worker")

COMMAND_BY_JOB_TYPE = {
    UpdateJob.JOB_UPDATE: "update",
    UpdateJob.JOB_LATEST_AGILE: "latest_agile",
}

# How often to evaluate whether a catch-up is due. This is not how often one can
# be queued — that is bounded by UPDATE_CATCHUP_HOURS, because queueing resets
# the very timestamp the check measures against.
CATCHUP_CHECK_SECONDS = 600


def maybe_enqueue_catchup(now=None):
    """Queue an update job if the external scheduler has missed a cycle.

    Production updates come from EasyCron, which lives outside this repo, fires
    once and never retries. When the site is unreachable at the moment it fires,
    the POST simply fails and that forecast cycle is lost with nothing to notice
    it (GH #104). This is the backstop for that, and for EasyCron stopping
    altogether — a failure that has previously gone unnoticed for three days.

    Returns the queued job, or None when no catch-up is warranted.
    """
    if not getattr(settings, "UPDATE_CATCHUP_ENABLED", True):
        return None

    hours = getattr(settings, "UPDATE_CATCHUP_HOURS", 9)
    if hours <= 0:
        return None

    now = now or timezone.now()

    # A job already queued or running means the pipeline is moving; there is
    # nothing to cover for, and enqueueing here would just duplicate work.
    if UpdateJob.objects.filter(
        job_type=UpdateJob.JOB_UPDATE,
        status__in=[UpdateJob.STATUS_PENDING, UpdateJob.STATUS_RUNNING],
    ).exists():
        return None

    latest = (
        UpdateJob.objects.filter(job_type=UpdateJob.JOB_UPDATE)
        .order_by("-requested_at")
        .first()
    )
    # A database with no update history has no missed cadence to recover.
    if latest is None:
        return None

    age = now - latest.requested_at
    if age < timedelta(hours=hours):
        return None

    # Measuring requested_at rather than finished_at, and matching the options
    # of the scheduled path exactly: run_job passes options straight into
    # call_command, so an unrecognised key would fail the job it is rescuing.
    job = UpdateJob.objects.create(
        job_type=UpdateJob.JOB_UPDATE, options={"skip_kde_plot": True}
    )
    logger.warning(
        "No update job for %.1fh (threshold %sh) — external scheduler appears to "
        "have missed a cycle; queued catch-up job id=%s",
        age.total_seconds() / 3600,
        hours,
        job.id,
    )
    return job


class Command(BaseCommand):
    help = "Poll for pending update jobs and run them outside the web process."

    def add_arguments(self, parser):
        parser.add_argument("--poll-interval", type=int, default=5)
        parser.add_argument("--once", action="store_true")

    def handle(self, *args, **options):
        poll_interval = options["poll_interval"]
        run_once = options["once"]

        logger.info("Starting update worker")
        self.retry_database_operation(self.fail_interrupted_jobs, poll_interval, run_once)

        # Deadline of 0 means the first pass checks immediately, so a worker
        # coming back after an outage notices a missed cycle at once rather than
        # ten minutes later.
        next_catchup_check = 0.0
        while True:
            close_old_connections()

            if time.monotonic() >= next_catchup_check:
                next_catchup_check = time.monotonic() + CATCHUP_CHECK_SECONDS
                self.retry_database_operation(maybe_enqueue_catchup, poll_interval, run_once)

            job = self.retry_database_operation(self.claim_job, poll_interval, run_once)
            if job is None:
                if run_once:
                    logger.info("No pending update jobs")
                    return
                time.sleep(poll_interval)
                continue

            self.run_job(job)
            if run_once:
                return

    def retry_database_operation(self, operation, poll_interval, run_once):
        while True:
            try:
                return operation()
            except DatabaseError:
                close_old_connections()
                logger.exception("Database unavailable while running update worker; retrying")
                if run_once:
                    raise
                time.sleep(poll_interval)

    def fail_interrupted_jobs(self):
        now = timezone.now()
        count = UpdateJob.objects.filter(status=UpdateJob.STATUS_RUNNING).update(
            status=UpdateJob.STATUS_FAILED,
            finished_at=now,
            error="Worker restarted before this job completed.",
        )
        if count:
            logger.warning("Marked %s interrupted worker job(s) as failed", count)

    def claim_job(self):
        with transaction.atomic():
            job = (
                UpdateJob.objects.select_for_update(skip_locked=True)
                .filter(status=UpdateJob.STATUS_PENDING)
                .order_by("requested_at")
                .first()
            )
            if job is None:
                return None

            job.status = UpdateJob.STATUS_RUNNING
            job.started_at = timezone.now()
            job.error = ""
            job.save(update_fields=["status", "started_at", "error"])
            return job

    def run_job(self, job):
        command_name = COMMAND_BY_JOB_TYPE.get(job.job_type)
        if command_name is None:
            message = f"Unknown worker job type: {job.job_type}"
            self.mark_job_failed(job, message)
            logger.error("Worker job %s failed: %s", job.id, message)
            return

        command_options = job.options if job.job_type == UpdateJob.JOB_UPDATE else {}
        logger.info(
            "Running worker job id=%s type=%s command=%s options=%s",
            job.id,
            job.job_type,
            command_name,
            command_options,
        )
        log_path = BASE_DIR / "logs" / "update_jobs" / f"{job.job_type}_job_{job.id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.set_job_log_file(job, log_path)

        old_update_log_file = os.environ.get("UPDATE_LOG_FILE")
        old_update_log_to_console = os.environ.get("UPDATE_LOG_TO_CONSOLE")
        os.environ["UPDATE_LOG_FILE"] = str(log_path)
        os.environ["UPDATE_LOG_TO_CONSOLE"] = "0"

        try:
            close_old_connections()
            with log_path.open("a") as log_file:
                with redirect_stdout(log_file), redirect_stderr(log_file):
                    call_command(command_name, **command_options)
        except Exception as exc:
            close_old_connections()
            self.append_to_job_log(log_path, traceback.format_exc())
            self.mark_job_failed(job, f"{exc}\n\n{traceback.format_exc()}")
            logger.exception("Worker job id=%s type=%s failed", job.id, job.job_type)
            return
        finally:
            if old_update_log_file is None:
                os.environ.pop("UPDATE_LOG_FILE", None)
            else:
                os.environ["UPDATE_LOG_FILE"] = old_update_log_file

            if old_update_log_to_console is None:
                os.environ.pop("UPDATE_LOG_TO_CONSOLE", None)
            else:
                os.environ["UPDATE_LOG_TO_CONSOLE"] = old_update_log_to_console

        close_old_connections()
        self.mark_job_completed(job)
        logger.info("Worker job id=%s type=%s completed", job.id, job.job_type)

    def set_job_log_file(self, job, log_path):
        while True:
            try:
                job.log_file = str(log_path)
                job.save(update_fields=["log_file"])
                return
            except DatabaseError:
                close_old_connections()
                logger.exception("Database unavailable while setting worker job log file; retrying")
                time.sleep(5)

    def append_to_job_log(self, log_path, content):
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a") as log_file:
            log_file.write("\n")
            log_file.write(content)

    def mark_job_failed(self, job, error):
        while True:
            try:
                job.status = UpdateJob.STATUS_FAILED
                job.finished_at = timezone.now()
                job.error = error
                job.save(update_fields=["status", "finished_at", "error"])
                return
            except DatabaseError:
                close_old_connections()
                logger.exception("Database unavailable while marking worker job failed; retrying")
                time.sleep(5)

    def mark_job_completed(self, job):
        while True:
            try:
                job.status = UpdateJob.STATUS_COMPLETED
                job.finished_at = timezone.now()
                job.save(update_fields=["status", "finished_at"])
                return
            except DatabaseError:
                close_old_connections()
                logger.exception("Database unavailable while marking worker job completed; retrying")
                time.sleep(5)
