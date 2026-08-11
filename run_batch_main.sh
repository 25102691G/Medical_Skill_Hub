#!/usr/bin/env bash
set -euo pipefail

# Gastroenterology Medical Diagnosis Pipeline
# This script runs the medical diagnosis pipeline with specified parameters

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

INPUT="database/sample5_test_nobhc.csv"
LIMIT=75
WORKERS=10
MODEL="${DIAGNOSIS_PROVIDER:-}"

# Run the Python script
python batch_main.py \
    --model "$MODEL" \
    --openai_apikey "${OPENAI_API_KEY:-}" \
    --openai_model "${OPENAI_MODEL:-}" \
    --deepseek_apikey "${DEEPSEEK_API_KEY:-}" \
    --deepseek_model "${DEEPSEEK_MODEL:-}" \
    --input "$INPUT" \
    --limit "$LIMIT" \
    --workers "$WORKERS"
