#!/bin/bash
set -euo pipefail

# Medical Guideline Skill Compilation Pipeline
# This script compiles all PDFs in a directory into local skill directories

INPUT_PDFS="./guidelines"
SKILLS_DIR="./skills"
SKILL_COMPILER_PROVIDER="deepseek"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
DEEPSEEK_BASE_URL="https://api.deepseek.com"
MINERU_DEVICE_MODE="${MINERU_DEVICE_MODE:-cpu}"
MINERU_COMMAND=".venv/bin/mineru -p {input} -o {output} -b pipeline -m auto -l ch"

export SKILL_COMPILER_PROVIDER
export DEEPSEEK_API_KEY
export DEEPSEEK_BASE_URL
export MINERU_DEVICE_MODE

.venv/bin/python compile_skill.py \
    --pdfs "$INPUT_PDFS" \
    --skills-dir "$SKILLS_DIR" \
    --mineru-command "$MINERU_COMMAND"
