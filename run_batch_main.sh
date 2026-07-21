#!/bin/bash
# Gastroenterology Medical Diagnosis Pipeline
# This script runs the medical diagnosis pipeline with specified parameters

INPUT="database/mimic_iv_test_case.csv"
LIMIT=1
OPENAI_MODEL="gpt-5.5"
DIAGNOSIS_TOPK=5

export OPENAI_MODEL
export DIAGNOSIS_TOPK

# Run the Python script
python batch_main.py \
    --input "$INPUT" \
    --limit "$LIMIT"
