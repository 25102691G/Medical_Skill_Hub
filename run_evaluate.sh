#!/usr/bin/env bash

INPUT="output/baseline/mimic_iv_llm_baseline_results_20260721_112615_232120_deepseek-v4-pro.jsonl"
MODEL=openai
WORKERS=50

python evaluate.py \
    --input "$INPUT" \
    --model "$MODEL" \
    --workers "$WORKERS"
