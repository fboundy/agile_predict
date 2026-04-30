from contextlib import redirect_stderr, redirect_stdout
import logging
import os
from pathlib import Path
import time
import traceback

from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import DatabaseError, close_old_connections, transaction
from django.utils import timezone

from config.settings import BASE_DIR
from prices.models import UpdateJob


logger = logging.getLogger(__name__)


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
        while True:
            close_old_connections()
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
            logger.warning("Marked %s interrupted update job(s) as failed", count)

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
        logger.info("Running update job %s with options %s", job.id, job.options)
        log_path = BASE_DIR / "logs" / "update_jobs" / f"update_job_{job.id}.log"
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
                    call_command("update", **job.options)
        except Exception as exc:
            close_old_connections()
            self.append_to_job_log(log_path, traceback.format_exc())
            self.mark_job_failed(job, f"{exc}\n\n{traceback.format_exc()}")
            logger.exception("Update job %s failed", job.id)
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
        logger.info("Update job %s completed", job.id)

    def set_job_log_file(self, job, log_path):
        while True:
            try:
                job.log_file = str(log_path)
                job.save(update_fields=["log_file"])
                return
            except DatabaseError:
                close_old_connections()
                logger.exception("Database unavailable while setting update job log file; retrying")
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
                logger.exception("Database unavailable while marking update job failed; retrying")
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
                logger.exception("Database unavailable while marking update job completed; retrying")
                time.sleep(5)
