#!/usr/bin/env bash
# Serve an OCR model with vLLM for the benchmark (OpenAI API on 127.0.0.1:$PORT/v1).
#
#   scripts/serve_vllm.sh <hf-model-id> <served-name> [extra vllm serve args...]
#
# Prefix caching and the multimodal processor cache are disabled: benchmark
# pages repeat across warm-up and concurrency sweeps, and cache hits would
# hide the real prefill cost of each page image.
#
# Environment overrides: VLLM_BIN, GPU, PORT, GPU_MEM_UTIL, MAX_MODEL_LEN, MAX_NUM_SEQS.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <hf-model-id> <served-name> [extra vllm serve args...]" >&2
  exit 2
fi
MODEL=$1
NAME=$2
shift 2

VLLM_BIN=${VLLM_BIN:-$HOME/Curator/.venv/bin/vllm}
GPU=${GPU:-1}
PORT=${PORT:-8010}
# Fraction of the GPU's *total* memory vLLM may use (weights + activations + KV cache).
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}

export CUDA_VISIBLE_DEVICES=$GPU
# The venv's bin holds ninja, which FlashInfer JIT builds look up on PATH.
export PATH="$(dirname "$VLLM_BIN"):$PATH"
# FlashInfer's sampler is JIT-compiled and the system nvcc (12.2) is older than
# this vLLM's CUDA (12.9). Use the PyTorch sampler; decoding is greedy anyway.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$NAME" \
  --host 127.0.0.1 --port "$PORT" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0 \
  "$@"
