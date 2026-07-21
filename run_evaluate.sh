#!/bin/bash
# Gastroenterology Medical Diagnosis Pipeline
# This script runs the medical diagnosis pipeline with specified parameters

# Run the Python script
python evaluate_diagnosis_results.py \
    --input "output/mimiv_iv_llm_baseline_results_20260720_175629_201688.jsonl"