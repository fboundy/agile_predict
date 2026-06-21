#!/usr/bin/env bash
# Reads UPDATE_TOKEN from a .env file and prints it, stripping surrounding quotes.
# Usage: TOKEN="$(read_env_token /path/to/.env)"
read_env_token() {
    local env_file="$1"
    ENV_FILE="$env_file" python3 - <<'PY'
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
}
