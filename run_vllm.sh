#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv-qwen/bin/activate"
set -a
# shellcheck disable=SC1091
source "$PROJECT_ROOT/.env"
set +a

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7
exec vllm serve "$QWEN_MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 131072 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --language-model-only \
    --tensor-parallel-size 4
