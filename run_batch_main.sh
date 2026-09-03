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

INPUT="database/mimic_test.csv"
LIMIT=2000
# 不同模型的 workers推荐值
# openai-gpt-5.5 
# deepseek-v4-pro: 50
# qwen3.5-27b: 40
# qwen3.5-122b-a10b: 25
WORKERS=40
# HISTORY_OUTPUT="output/batch/mimic_test_2000_20260903_110705_298959.jsonl"
MODEL="${DIAGNOSIS_PROVIDER:-}"

HISTORY_ARGS=()
if [[ -n "${HISTORY_OUTPUT:-}" ]]; then
    HISTORY_ARGS=(--history-output "$HISTORY_OUTPUT")
fi

# Run the Python script
"$PROJECT_ROOT/.venv/bin/python" batch_main.py \
    --model "$MODEL" \
    --openai_apikey "${OPENAI_API_KEY:-}" \
    --deepseek_apikey "${DEEPSEEK_API_KEY:-}" \
    --input "$INPUT" \
    --limit "$LIMIT" \
    --workers "$WORKERS" \
    "${HISTORY_ARGS[@]}"
