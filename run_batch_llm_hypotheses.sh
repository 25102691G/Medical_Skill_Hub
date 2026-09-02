#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

INPUT="database/mimic_test.csv"
LIMIT=2000
WORKERS=10
BASE_URL="http://127.0.0.1:8000/v1"
# MODEL=diagagent-local
MODEL=baichuan-m2-local

"$PROJECT_ROOT/.venv/bin/python" batch_llm_hypotheses.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --limit "$LIMIT" \
    --workers "$WORKERS"
