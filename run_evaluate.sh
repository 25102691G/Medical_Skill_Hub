#!/usr/bin/env bash

INPUT="output/baseline/mimic_iv_llm_baseline_results_20260721_133355_701343_deepseek-v4-pro.jsonl"
MODEL=deepseek
WORKERS=50

python evaluate.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --workers "$WORKERS"
