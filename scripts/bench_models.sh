#!/usr/bin/env bash
# Benchmark several models one after another on GPU $GPU: start each model's
# vLLM server, run the concurrency sweep, stop the server.
#
#   scripts/bench_models.sh [model-key ...]     # default: all keys below
#
# Results: results/<key>_sweep/, server logs: logs/vllm_<key>.log.
# Environment overrides: GPU, PORT, CONCURRENCY, SCOPE, and those of serve_vllm.sh.
# DELETE_WEIGHTS=1 removes a Hub model's downloaded weights after its sweep.
set -uo pipefail

cd "$(dirname "$0")/.."
export GPU=${GPU:-1}
export PORT=${PORT:-8010}
CONCURRENCY=${CONCURRENCY:-1,4,8,16}
SCOPE=${SCOPE:-scored}
DELETE_WEIGHTS=${DELETE_WEIGHTS:-0}
HF_BIN=${HF_BIN:-$HOME/Curator/.venv/bin/hf}

# Per model key:
#   MODEL       Hub id or local dir
#   RUN_ARGS    ocrbench run arguments (whitespace-separated, no quoting)
#   BODY        JSON merged into every request (--extra-body)
#   SERVE_ARGS  extra vllm serve arguments (whitespace-separated, no quoting)
#   SERVE_ENV   VAR=value overrides for serve_vllm.sh
declare -A MODEL RUN_ARGS BODY SERVE_ARGS SERVE_ENV

MODEL[lightonocr-2-1b]=lightonai/LightOnOCR-2-1B
RUN_ARGS[lightonocr-2-1b]="--prompt lightonocr"

# Tokenizer config converted for transformers v4: scripts/prepare_surya.py.
MODEL[surya-ocr-2]=models/surya-ocr-2
RUN_ARGS[surya-ocr-2]="--prompt surya --max-tokens 8192"

# n-gram repetition blocker from the vLLM DeepSeek-OCR recipe; <td>, </td> may repeat.
MODEL[deepseek-ocr-2]=deepseek-ai/DeepSeek-OCR-2
RUN_ARGS[deepseek-ocr-2]="--prompt deepseek-ocr"
BODY[deepseek-ocr-2]='{"skip_special_tokens": false, "vllm_xargs": {"ngram_size": 30, "window_size": 90, "whitelist_token_ids": [128821, 128822]}}'
SERVE_ARGS[deepseek-ocr-2]="--logits-processors vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor"
SERVE_ENV[deepseek-ocr-2]="MAX_MODEL_LEN=8192"  # model's max_position_embeddings

MODEL[qwen2.5-vl-3b]=Qwen/Qwen2.5-VL-3B-Instruct
RUN_ARGS[qwen2.5-vl-3b]="--prompt generic"

MODEL[qwen3-vl-4b]=Qwen/Qwen3-VL-4B-Instruct
RUN_ARGS[qwen3-vl-4b]="--prompt generic"

MODEL[paddleocr-vl-1.5]=PaddlePaddle/PaddleOCR-VL-1.5
RUN_ARGS[paddleocr-vl-1.5]="--prompt paddleocr-vl"
SERVE_ARGS[paddleocr-vl-1.5]="--trust-remote-code"

# Sampling from the HunyuanOCR model card.
MODEL[hunyuan-ocr]=tencent/HunyuanOCR
RUN_ARGS[hunyuan-ocr]="--prompt hunyuan-ocr"
BODY[hunyuan-ocr]='{"top_k": -1, "repetition_penalty": 1.08, "skip_special_tokens": true}'
SERVE_ARGS[hunyuan-ocr]="--trust-remote-code"
# HunyuanOCR-1.5's config stores xdrope under the v5 key rope_parameters.
SERVE_ENV[hunyuan-ocr]="PYTHONPATH=$PWD/.tf5"

MODEL[dots-ocr]=dots-studio/dots.ocr
RUN_ARGS[dots-ocr]="--prompt dots-ocr"
SERVE_ARGS[dots-ocr]="--trust-remote-code --chat-template-content-format string"

MODEL[mineru2.5]=opendatalab/MinerU2.5-2509-1.2B
RUN_ARGS[mineru2.5]="--prompt mineru"

# glm_ocr configs need transformers v5: scripts/prepare_tf5.sh installs it
# into .tf5/, which is put first on PYTHONPATH for these servers only.
MODEL[glm-ocr]=zai-org/GLM-OCR
RUN_ARGS[glm-ocr]="--prompt glm-ocr"
SERVE_ENV[glm-ocr]="PYTHONPATH=$PWD/.tf5"

ORDER=(lightonocr-2-1b surya-ocr-2 deepseek-ocr-2 qwen2.5-vl-3b qwen3-vl-4b
       paddleocr-vl-1.5 hunyuan-ocr dots-ocr mineru2.5 glm-ocr)

server_pid=
stop_server() {
  [ -n "$server_pid" ] || return 0
  kill "$server_pid" 2>/dev/null
  for _ in $(seq 60); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
  kill -9 "$server_pid" 2>/dev/null
  server_pid=
}
trap stop_server EXIT

is_hub_id() { [[ "$1" == */* && ! -e "$1" ]]; }

mkdir -p logs results
status=0
for key in "${@:-${ORDER[@]}}"; do
  model=${MODEL[$key]:-}
  if [ -z "$model" ]; then
    echo "### $key: unknown model key" >&2; status=1; continue
  fi
  if is_hub_id "$model"; then
    echo "### $key: downloading $model ($(date +%T))"
    if ! "$HF_BIN" download "$model" > /dev/null 2>> "logs/download_$key.log"; then
      echo "### $key: download FAILED, see logs/download_$key.log" >&2; status=1; continue
    fi
  fi

  echo "### $key: starting server ($(date +%T))"
  # shellcheck disable=SC2086  # SERVE_ENV / SERVE_ARGS hold several words
  env ${SERVE_ENV[$key]:-} scripts/serve_vllm.sh "$model" "$key" ${SERVE_ARGS[$key]:-} \
    > "logs/vllm_$key.log" 2>&1 &
  server_pid=$!
  ready=
  for _ in $(seq 180); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; then ready=1; break; fi
    kill -0 "$server_pid" 2>/dev/null || break
    sleep 5
  done
  if [ -z "$ready" ]; then
    echo "### $key: server failed to start, see logs/vllm_$key.log" >&2
    stop_server; status=1; continue
  fi

  echo "### $key: server ready, running sweep $CONCURRENCY ($(date +%T))"
  body=()
  [ -n "${BODY[$key]:-}" ] && body=(--extra-body "${BODY[$key]}")
  # shellcheck disable=SC2086  # RUN_ARGS holds several words
  if uv run ocrbench run --gpus "$GPU" --base-url "http://localhost:$PORT/v1" \
      --scope "$SCOPE" --concurrency "$CONCURRENCY" --name "${key}_sweep" \
      ${RUN_ARGS[$key]} "${body[@]}" > "logs/${key}_sweep.log" 2>&1; then
    echo "### $key: done ($(date +%T))"
  else
    echo "### $key: benchmark FAILED, see logs/${key}_sweep.log" >&2; status=1
  fi
  stop_server

  if [ "$DELETE_WEIGHTS" = 1 ] && is_hub_id "$model"; then
    rm -rf "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${model//\//--}"
    echo "### $key: deleted downloaded weights"
  fi
done
exit $status
