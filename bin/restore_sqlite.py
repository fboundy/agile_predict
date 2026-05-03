import csv
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BACKUP = ROOT_DIR / ".local" / "backup.sql"
DEFAULT_SQLITE = ROOT_DIR / "db.sqlite3"

TABLES = {
    "auth_group",
    "auth_group_permissions",
    "auth_permission",
    "auth_user",
    "auth_user_groups",
    "auth_user_user_permissions",
    "django_admin_log",
    "django_content_type",
    "django_migrations",
    "django_session",
    "prices_agiledata",
    "prices_forecastdata",
    "prices_forecasts",
    "prices_history",
    "prices_pricehistory",
}


def parse_copy_header(line):
    prefix = "COPY public."
    if not line.startswith(prefix) or " FROM stdin;" not in line:
        return None
    table, columns = line[len(prefix) :].split(" (", 1)
    columns = columns.split(") FROM stdin;", 1)[0]
    return table, [column.strip() for column in columns.split(",")]


def parse_copy_value(value):
    if value == r"\N":
        return None
    return value.replace(r"\t", "\t").replace(r"\n", "\n").replace(r"\\", "\\")


def import_copy_sections(sql_path, sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = OFF")

    current = None
    inserted = {}
    with sql_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if current is None:
                header = parse_copy_header(line)
                if header is None:
                    continue
                table, columns = header
                if table not in TABLES:
                    current = ("skip", table, columns, None)
                    continue
                placeholders = ",".join(["?"] * len(columns))
                quoted_columns = ",".join([f'"{column}"' for column in columns])
                statement = f'INSERT INTO "{table}" ({quoted_columns}) VALUES ({placeholders})'
                conn.execute(f'DELETE FROM "{table}"')
                current = ("import", table, columns, statement)
                inserted[table] = 0
                continue

            mode, table, columns, statement = current
            if line == r"\.":
                current = None
                continue
            if mode == "skip":
                continue

            values = next(csv.reader([line], delimiter="\t", escapechar=None, quoting=csv.QUOTE_NONE))
            conn.execute(statement, [parse_copy_value(value) for value in values])
            inserted[table] += 1

    conn.commit()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.close()
    return inserted


def run_migrations(sqlite_path):
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{sqlite_path}"
    env["SECRET_KEY"] = "restore-sqlite"
    env["DEBUG"] = "false"
    subprocess.run([sys.executable, "manage.py", "migrate", "--noinput"], cwd=ROOT_DIR, env=env, check=True)


def main():
    sql_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_BACKUP
    sqlite_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_SQLITE

    if not sql_path.exists():
        raise SystemExit(f"Backup file not found: {sql_path}")

    if sqlite_path.exists():
        sqlite_path.unlink()

    print(f"Creating SQLite database: {sqlite_path}")
    run_migrations(sqlite_path)

    print(f"Importing PostgreSQL COPY data from: {sql_path}")
    inserted = import_copy_sections(sql_path, sqlite_path)

    print("Imported rows:")
    for table in sorted(inserted):
        print(f"  {table}: {inserted[table]}")
    print("Restore completed!")


if __name__ == "__main__":
    main()
