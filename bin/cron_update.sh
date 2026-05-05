#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
LOG_DIR="${ROOT_DIR}/logs"
URL="${UPDATE_CRON_URL:-http://127.0.0.1:8000/update?scheduled=1}"

mkdir -p "$LOG_DIR"

TOKEN="$(
    ENV_FILE="$ENV_FILE" python3 - <<'PY'
import os
from pathlib import Path

token = ""
for line in Path(os.environ["ENV_FILE"]).read_text(encoding="utf-8").splitlines():
    if not line.startswith("UPDATE_TOKEN="):
        continue
    token = line.split("=", 1)[1].strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1]

print(token)
PY
)"

if [ -z "$TOKEN" ]; then
    echo "UPDATE_TOKEN is not set in ${ENV_FILE}" >&2
    exit 1
fi

curl -fsS -X POST -H "X-Update-Token: ${TOKEN}" "$URL"
