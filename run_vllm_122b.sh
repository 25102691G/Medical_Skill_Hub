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
exec vllm serve "$PROJECT_ROOT/models/Qwen3.5-122B-A10B" \
    --served-model-name Qwen3.5-122B-A10B \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 64 \
    --performance-mode throughput \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --language-model-only \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --api-server-count 1 \
    --aggregate-engine-logging
