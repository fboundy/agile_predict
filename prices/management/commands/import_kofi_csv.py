import csv
from datetime import datetime, timezone as dt_timezone

from django.core.management.base import BaseCommand, CommandError

from prices.models import KofiPayment


class Command(BaseCommand):
    help = (
        "Import a Ko-fi transaction CSV export (Transaction_All.csv) into KofiPayment. "
        "Idempotent on TransactionId; skips outgoing ('Given') rows; does not store emails."
    )

    def add_arguments(self, parser):
        parser.add_argument("csv_path")
        parser.add_argument("--dry-run", action="store_true")

    def handle(self, *args, **options):
        path = options["csv_path"]
        created = existing = skipped = 0
        try:
            fh = open(path, newline="", encoding="utf-8-sig")
        except OSError as exc:
            raise CommandError(f"Cannot open {path}: {exc}")
        with fh:
            for row in csv.DictReader(fh):
                txn = (row.get("TransactionId") or "").strip()
                try:
                    received = float(row.get("Received") or 0)
                except ValueError:
                    received = 0
                if not txn or received <= 0:
                    skipped += 1  # outgoing/zero rows are not revenue
                    continue
                ts = datetime.strptime(
                    row["DateTime (UTC)"].strip(), "%m/%d/%Y %H:%M"
                ).replace(tzinfo=dt_timezone.utc)
                ttype = (row.get("TransactionType") or "").strip()
                if options["dry_run"]:
                    created += 1
                    continue
                _, was_created = KofiPayment.objects.get_or_create(
                    kofi_transaction_id=txn,
                    defaults={
                        "timestamp": ts,
                        "payment_type": ttype,
                        "from_name": (row.get("From") or "").strip(),
                        "message": (row.get("Message") or "").strip(),
                        "amount": received,
                        "currency": ((row.get("Currency") or "GBP").strip().upper() or "GBP"),
                        "is_public": True,
                        "is_subscription_payment": "monthly" in ttype.lower(),
                    },
                )
                if was_created:
                    created += 1
                else:
                    existing += 1
        self.stdout.write(
            self.style.SUCCESS(
                f"Imported {created} payment(s); {existing} already present; {skipped} outgoing/zero rows skipped"
            )
        )
