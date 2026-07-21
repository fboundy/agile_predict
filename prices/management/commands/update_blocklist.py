from django.core.management.base import BaseCommand

from prices import blocklist


class Command(BaseCommand):
    help = "Fetch and cache the reputation IP blocklist (see settings.BLOCKLIST_URLS)."

    def add_arguments(self, parser):
        parser.add_argument("--test-ip", help="Report whether an IP is currently blocked, then exit.")

    def handle(self, *args, **options):
        test_ip = options.get("test_ip")
        if test_ip:
            blocked = blocklist.is_blocked(test_ip)
            self.stdout.write(f"{test_ip}: {'BLOCKED' if blocked else 'allowed'} (entries={blocklist.status()['count']})")
            return

        count = blocklist.refresh_now()
        if count:
            self.stdout.write(self.style.SUCCESS(f"Blocklist refreshed: {count} IPv4 ranges"))
        else:
            self.stdout.write(self.style.WARNING("Blocklist refresh returned no entries (fetch failed?) — leaving current list unchanged"))
