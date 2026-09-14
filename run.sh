#!/usr/bin/env bash
# ── Taurus Dashboard — lancement ────────────────────────────────────────
# Usage : ./run.sh [port]
set -euo pipefail

cd "$(dirname "$0")"
PORT="${1:-8000}"

# Charge .env s'il existe (clés d'API, User-Agent SEC, réglages du cache).
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

echo "Taurus Dashboard → http://127.0.0.1:${PORT}"
exec python3 -m uvicorn backend.app:app --host 127.0.0.1 --port "${PORT}" "$@"
