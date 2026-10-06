#!/usr/bin/env bash
# Install transformers v5 into .tf5/ as an overlay for the vLLM venv, for
# models whose configs only exist in v5 (GLM-OCR). The vLLM venv is untouched;
# bench_models.sh puts .tf5 first on PYTHONPATH for those servers only.
#
# vLLM 0.22 accepts transformers >=5.5.1; 5.13.1 is from shortly before it.
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM_PY=${VLLM_PY:-$HOME/Curator/.venv/bin/python}

uv pip install --python "$VLLM_PY" --target .tf5 --no-deps \
  "transformers==5.13.1" "huggingface-hub>=1.5,<2" "safetensors>=0.8"

PYTHONPATH=$PWD/.tf5 "$VLLM_PY" - <<'EOF'
import huggingface_hub, transformers
from transformers import AutoConfig
print("transformers", transformers.__version__, "| huggingface_hub", huggingface_hub.__version__)
print(type(AutoConfig.from_pretrained("zai-org/GLM-OCR")).__name__)
EOF
