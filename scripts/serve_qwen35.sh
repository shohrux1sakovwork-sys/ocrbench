#!/usr/bin/env bash
# Serve Qwen3.5-9B with vLLM for the benchmark. See serve_vllm.sh for settings.
exec "$(dirname "$0")/serve_vllm.sh" Qwen/Qwen3.5-9B qwen3.5-9b --reasoning-parser qwen3 "$@"
