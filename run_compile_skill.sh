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
MINERU_DEVICE_MODE="${MINERU_DEVICE_MODE:-cuda}"
NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
MINERU_COMMAND="${MINERU_COMMAND:-}"
if [[ -z "$MINERU_COMMAND" ]]; then
    MINERU_COMMAND='.venv/bin/mineru -p {input} -o {output} -b pipeline -m auto -l ch'
fi

export SKILL_COMPILER_PROVIDER
export DEEPSEEK_API_KEY
export DEEPSEEK_BASE_URL
export DEEPSEEK_MODEL
export MINERU_DEVICE_MODE
export NO_PROXY
export no_proxy

if [[ $# -gt 0 ]]; then
    INPUT_ARGS=("$@")
else
    INPUT_ARGS=(--pdfs "$INPUT_PDFS" --workers 10)
fi

.venv/bin/python compile_skill.py \
    "${INPUT_ARGS[@]}" \
    --skills-dir "$SKILLS_DIR" \
    --mineru-command "$MINERU_COMMAND"
