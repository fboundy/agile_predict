#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
LOG_DIR="${ROOT_DIR}/logs"
# feature_set pinned to the live extreme-detection re-scoring candidate
# (bm_wind + opmr_national_surplus on top of full_fr) — see
# docs/MODEL_DYNAMIC_RANGE.md, 2026-09-11. Remove this param to fall back to
# the stored feature-experiment winner.
URL="${UPDATE_CRON_URL:-http://127.0.0.1:8000/update?scheduled=1&feature_set=full_fr_bm_opmr}"

mkdir -p "$LOG_DIR"

# shellcheck source=lib/read_token.sh
. "${SCRIPT_DIR}/lib/read_token.sh"
TOKEN="$(read_env_token "$ENV_FILE")"

if [ -z "$TOKEN" ]; then
    echo "UPDATE_TOKEN is not set in ${ENV_FILE}" >&2
    exit 1
fi

curl -fsS -X POST -H "X-Update-Token: ${TOKEN}" "$URL"
