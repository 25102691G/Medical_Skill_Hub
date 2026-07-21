#!/usr/bin/env bash

INPUT="output/similar_case/similar_case_results_20260721_175316_699909.jsonl"
MODEL=deepseek
WORKERS=50

python evaluate_similar_case.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --workers "$WORKERS"
