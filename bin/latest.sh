#!/usr/bin/env sh
set -eu

: "${UPDATE_TOKEN:?UPDATE_TOKEN is required}"

date
curl -i -X POST "https://agilepredict.com/update/latest_agile" \
  -H "X-Update-Token: ${UPDATE_TOKEN}"
date
