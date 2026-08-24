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
exec vllm serve "$PROJECT_ROOT/models/Qwen3.5-9B" \
    --served-model-name qwen-local \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --performance-mode throughput \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --language-model-only \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --api-server-count 4 \
    --aggregate-engine-logging
