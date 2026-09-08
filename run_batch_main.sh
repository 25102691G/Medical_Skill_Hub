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

INPUT="database/mimic_test_random_200.csv"
LIMIT=49
# 不同模型的 workers推荐值
# openai-gpt-5.5 
# deepseek-v4-pro: 50
# qwen3.5-27b: 40
# qwen3.5-122b-a10b: 15
WORKERS=100
HISTORY_OUTPUT="output/batch/deepseek-v4-pro_mimic_test_random_200_1_20260908_124706_724740.jsonl"
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
