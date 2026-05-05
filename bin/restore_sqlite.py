import csv
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BACKUP = ROOT_DIR / ".local" / "backup.sql"
DEFAULT_SQLITE = ROOT_DIR / "db.sqlite3"


def increase_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = limit // 10


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
    if value.startswith(r"\x"):
        try:
            return bytes.fromhex(value[2:])
        except ValueError:
            pass
    return value.replace(r"\t", "\t").replace(r"\n", "\n").replace(r"\\", "\\")


def get_sqlite_tables(conn):
    rows = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        """
    )
    return {row[0] for row in rows}


def reset_sqlite_sequence(conn, table):
    sequence_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'"
    ).fetchone()
    if not sequence_exists:
        return

    columns = {
        row[1]
        for row in conn.execute(f'PRAGMA table_info("{table}")')
    }
    if "id" not in columns:
        return

    max_id = conn.execute(f'SELECT MAX(id) FROM "{table}"').fetchone()[0]
    if max_id is not None:
        conn.execute("UPDATE sqlite_sequence SET seq = ? WHERE name = ?", (max_id, table))


def import_copy_sections(sql_path, sqlite_path):
    conn = sqlite3.connect(sqlite_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    tables = get_sqlite_tables(conn)

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
                if table not in tables:
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

    for table in inserted:
        reset_sqlite_sequence(conn, table)

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

    env_database_url = os.environ.get("DATABASE_URL", "")
    expected_database_url = f"sqlite:///{sqlite_path}"
    if env_database_url and env_database_url != expected_database_url:
        raise SystemExit(
            "DATABASE_URL must point at the SQLite database being restored.\n"
            f"Expected: {expected_database_url}\n"
            f"Found:    {env_database_url}"
        )
    os.environ["DATABASE_URL"] = expected_database_url

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
    increase_csv_field_limit()
    main()
