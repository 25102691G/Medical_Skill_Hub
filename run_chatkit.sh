#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

export DIAGNOSIS_PROVIDER="${DIAGNOSIS_PROVIDER:-}"
exec .venv/bin/uvicorn chatkit_app.app:app --host 0.0.0.0 --port 8000 --reload
