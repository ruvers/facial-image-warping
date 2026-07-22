#!/usr/bin/env bash
# Run API + static UI from the repository root (fixes "No module named 'backend'").
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ -f .venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
elif [[ -f backend/.venv/bin/activate ]]; then
  # shellcheck source=/dev/null
  source backend/.venv/bin/activate
fi

exec uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
