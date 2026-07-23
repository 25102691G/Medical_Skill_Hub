#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: bash run_evaluate.sh <batch-results.jsonl>" >&2
    exit 1
fi

INPUT="$1"
MODEL=deepseek
WORKERS=50

python evaluate.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --workers "$WORKERS"
