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

INPUT="${INPUT:-}"
LIMIT=10000
MODEL=deepseek

python llm_baseline.py \
    --model "$MODEL" \
    --openai_apikey "${OPENAI_API_KEY:-}" \
    --openai_model "${OPENAI_MODEL:-}" \
    --deepseek_apikey "${DEEPSEEK_API_KEY:-}" \
    --deepseek_model "${DEEPSEEK_MODEL:-}" \
    --input "$INPUT" \
    --limit "$LIMIT"
