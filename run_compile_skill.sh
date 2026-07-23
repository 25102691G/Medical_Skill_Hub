#!/bin/bash
set -euo pipefail

# Medical Guideline Skill Compilation Pipeline
# This script compiles all PDFs in a directory into local skill directories

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.env"
    set +a
fi

INPUT_PDFS="${INPUT_PDFS:-./guidelines}"
SKILLS_DIR="${SKILLS_DIR:-./skills}"
SKILL_COMPILER_PROVIDER="${SKILL_COMPILER_PROVIDER:-deepseek}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
MINERU_DEVICE_MODE="${MINERU_DEVICE_MODE:-cpu}"
MINERU_COMMAND="${MINERU_COMMAND:-.venv/bin/mineru -p {input} -o {output} -b pipeline -m auto -l ch}"

export SKILL_COMPILER_PROVIDER
export DEEPSEEK_API_KEY
export DEEPSEEK_BASE_URL
export DEEPSEEK_MODEL
export MINERU_DEVICE_MODE

.venv/bin/python compile_skill.py \
    --pdfs "$INPUT_PDFS" \
    --skills-dir "$SKILLS_DIR" \
    --mineru-command "$MINERU_COMMAND" \
    --force
