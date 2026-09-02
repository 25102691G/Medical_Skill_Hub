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
export CUDA_VISIBLE_DEVICES=0,1,2,3
exec vllm serve "$PROJECT_ROOT/models/Baichuan-M2-32B" \
    --served-model-name baichuan-m2-local \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --performance-mode throughput \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"thinking_mode": "off"}' \
    --language-model-only \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --api-server-count 1 \
    --aggregate-engine-logging
