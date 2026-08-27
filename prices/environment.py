"""Host-environment detection, used to keep dev-only data off production.

The `History` backfill exists so plunge/spike behaviour can be studied across
seasons (see docs/MODEL_DYNAMIC_RANGE.md). The owner's decision on 2026-08-27 is
that it lives on the CT dev box only, so production's database stays small. The
backfill command therefore lives on the `dev` branch and is not merged to `main` —
but branches get merged and deployed by accident, so the prohibition is enforced
in code as well, not left to branch hygiene.

Two independent markers are checked, and *either* is enough to refuse:

* ``FLY_APP_NAME`` / ``FLY_MACHINE_ID`` — set automatically by fly.io on every
  machine. Present on production, absent on the CT.
* a non-SQLite default database — production runs fly Postgres
  (``DATABASE_URL``), the dev box runs local SQLite.

The two are genuinely complementary rather than belt-and-braces for its own sake.
``config/settings.py`` calls ``env.read_env(override=True)``, so a ``.env`` file
**overrides real environment variables** — a dev-style ``.env`` shipped onto a
production host would make the database marker read "sqlite" and miss. The fly
markers are injected by the platform into the process environment and are not
readable from, or overridable by, ``.env``. Conversely a non-fly production host
would have no fly markers but would still be caught by the database engine.

Verified against the live production machine on 2026-08-27: both markers fire
there (``FLY_APP_NAME``/``FLY_MACHINE_ID``/``FLY_ALLOC_ID`` set,
``django.db.backends.postgresql``), and the dev box trips neither.

The check deliberately **fails closed**: anything that is not positively
identifiable as the SQLite dev box is treated as production. A guard that guesses
"probably dev" when unsure is not a guard.
"""

import os

from django.conf import settings

# Any of these being set means we are on a fly.io machine.
FLY_MARKERS = ("FLY_APP_NAME", "FLY_MACHINE_ID", "FLY_ALLOC_ID")


def fly_markers_present():
    """Names of fly.io environment markers that are set (empty when not on fly)."""
    return [name for name in FLY_MARKERS if os.environ.get(name)]


def database_engine():
    try:
        return settings.DATABASES["default"].get("ENGINE", "")
    except Exception:  # pragma: no cover - settings always configured in practice
        return ""


def is_production():
    """True unless this is positively the local SQLite dev box.

    Fails closed: an unrecognised database engine, or any fly.io marker, counts as
    production.
    """
    if fly_markers_present():
        return True
    return "sqlite" not in database_engine()


def production_reasons():
    """Human-readable reasons is_production() returned True, for error messages."""
    reasons = []
    markers = fly_markers_present()
    if markers:
        reasons.append(f"fly.io environment markers set ({', '.join(markers)})")
    engine = database_engine()
    if "sqlite" not in engine:
        reasons.append(f"database engine is {engine or 'unknown'}, not SQLite")
    return reasons


def require_non_production(what):
    """Raise CommandError unless running on the dev box.

    `what` names the operation, for the error message. There is deliberately no
    override flag: bypassing this should require editing the code, which is a
    visible, reviewable act rather than a command-line typo.
    """
    from django.core.management.base import CommandError

    if is_production():
        reasons = production_reasons() or ["environment could not be identified as dev"]
        raise CommandError(
            f"{what} is disabled on production.\n"
            + "".join(f"  - {r}\n" for r in reasons)
            + "This data is held on the dev box only, to keep production's database\n"
            "small (see docs/MODEL_DYNAMIC_RANGE.md). If you genuinely need it on\n"
            "production, change prices/environment.py deliberately — there is no flag."
        )
