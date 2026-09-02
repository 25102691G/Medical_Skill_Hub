#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

INPUT="database/mimic_test.csv"
LIMIT=2000
WORKERS=200

"$PROJECT_ROOT/.venv/bin/python" rag_baseline.py \
    --input "$INPUT" \
    --limit "$LIMIT" \
    --workers "$WORKERS"
