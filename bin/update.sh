#!/usr/bin/env sh
set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"

if [ -z "${UPDATE_TOKEN:-}" ] && [ -f "${ROOT_DIR}/.env" ]; then
  UPDATE_TOKEN="$(
    sed -n 's/^[[:space:]]*UPDATE_TOKEN[[:space:]]*=[[:space:]]*//p' "${ROOT_DIR}/.env" \
      | tail -n 1 \
      | tr -d '\r' \
      | sed "s/^['\"]//; s/['\"]$//"
  )"
  export UPDATE_TOKEN
fi

: "${UPDATE_TOKEN:?UPDATE_TOKEN is required}"

date
curl -i -X POST "https://agilepredict.com/update?debug=true" \
  -H "X-Update-Token: ${UPDATE_TOKEN}"
date
