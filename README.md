# ocrbench

Benchmarks an OCR model served behind an OpenAI-compatible endpoint (vLLM,
SGLang, ...) on the PDFs in `test_uchun_ddoklar/` against `ground_truth.json`.

It reports:

- **Accuracy**: CER and WER, micro (total edits / total reference length) and per-page mean, plus substitution, insertion and deletion counts. The raw variants count Markdown/HTML markup as errors.
- **Time**: wall time, pages/s, output tokens/s, latency p50/p90/p95/p99, render time, and empirical complexity (log-log fit of latency against output tokens, image pixels and document pages).
- **Space**: peak RSS of the client and render workers, RSS growth per page, peak bytes of page images in flight, image size, prompt tokens, and GPU memory/utilization (via `nvidia-smi`).
- **Parallelism**: the effective number of requests in flight. A concurrency sweep adds speedup and efficiency per level.
- **Model (vLLM `/metrics`)**: KV-cache usage (peak and mean), requests running vs. waiting, preemptions, and server-side TTFT, inter-token latency, queue, prefill and decode time per request, plus prompt and generated tokens. Streaming requests add client-side TTFT and per-request decode speed.

## Ground truth

Only PDFs with an entry in `ground_truth.json` are opened. Matching uses file
names only: the ground truth stores sanitized names (`08-1-2224++.pdf` →
`08-1-2224__.pdf`, `ЎРҚ-1166.pdf` → `У_РҚ-1166.pdf`).

Entries with status `pending` (no text) or `raw_text_layer_not_ground_truth`
are never used as references. Their pages can still be run for timing and
memory with `--scope all`.

```bash
uv run ocrbench list --check-pages   # matched files, statuses, page counts
uv run ocrbench prune                # dry run: files with no ground truth
uv run ocrbench prune --yes          # delete them
```

## Model server

Models are served with vLLM on GPU1 (uses the vLLM 0.22 install in `~/Curator/.venv`):

```bash
scripts/serve_vllm.sh <hf-model-id-or-dir> <served-name> [vllm args]   # http://127.0.0.1:8010/v1
scripts/serve_qwen35.sh                                   # Qwen3.5-9B, model "qwen3.5-9b"
GPU_MEM_UTIL=0.5 MAX_NUM_SEQS=16 scripts/serve_qwen35.sh  # overrides
```

Prefix caching and the multimodal processor cache are off. Benchmark pages
repeat (warm-up, concurrency sweeps), and cached image prefixes would hide the
prefill cost. The report warns when the hit rate is high.

The FlashInfer sampler is disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`): it is
JIT-compiled and the system nvcc (CUDA 12.2) is older than vLLM's CUDA 12.9.
Decoding is greedy, so the PyTorch sampler gives the same tokens.

Qwen3.5 thinks by default. Requests send `chat_template_kwargs.enable_thinking=false`
(use `--thinking` to keep it on), and any `<think>` block is removed before
scoring.

## Comparing models

`scripts/bench_models.sh` starts each model's server, runs the 1/4/8/16 sweep
and stops the server, one model after another:

```bash
scripts/bench_models.sh                                  # all models below
scripts/bench_models.sh surya-ocr-2 lightonocr-2-1b      # some of them
CONCURRENCY=16 scripts/bench_models.sh qwen3-vl-4b       # other levels
uv run python scripts/compare_models.py                  # -> results/comparison.csv
```

| key | model | prompt (`--prompt`) | notes |
|---|---|---|---|
| `lightonocr-2-1b` | lightonai/LightOnOCR-2-1B | `lightonocr`: image only | |
| `surya-ocr-2` | datalab-to/surya-ocr-2 | `surya`: full-page HTML prompt | run `python scripts/prepare_surya.py` first; max 8192 tokens |
| `deepseek-ocr-2` | deepseek-ai/DeepSeek-OCR-2 | `deepseek-ocr`: `Free OCR.` | n-gram repetition blocker, 8192 context |
| `qwen2.5-vl-3b` | Qwen/Qwen2.5-VL-3B-Instruct | `generic` | |
| `qwen3-vl-4b` | Qwen/Qwen3-VL-4B-Instruct | `generic` | |
| `paddleocr-vl-1.5` | PaddlePaddle/PaddleOCR-VL-1.5 | `paddleocr-vl`: `OCR:` | whole page (normally layout crops) |
| `hunyuan-ocr` | tencent/HunyuanOCR | `hunyuan-ocr`: text extraction | repetition_penalty 1.08; needs `.tf5` |
| `dots-ocr` | dots-studio/dots.ocr | `dots-ocr`: `prompt_ocr` | skips page headers/footers by design |
| `mineru2.5` | opendatalab/MinerU2.5-2509-1.2B | `mineru`: `Text Recognition:` | whole page (normally layout crops) |
| `glm-ocr` | zai-org/GLM-OCR | `glm-ocr`: `Text Recognition:` | needs `.tf5`; whole page (normally layout crops) |

GLM-OCR and HunyuanOCR-1.5 configs need transformers v5: `scripts/prepare_tf5.sh`
installs transformers 5.13.1 into `.tf5/`, which `bench_models.sh` puts first on
`PYTHONPATH` for those servers only. `DELETE_WEIGHTS=1` deletes each Hub model's
weights after its sweep.

Surya 2's published tokenizer config is in the transformers v5 format;
`scripts/prepare_surya.py` links the downloaded snapshot into
`models/surya-ocr-2` with a v4 `tokenizer_config.json` (same tokenizer).

All models get the same page images (150 DPI, longest side 1800 px) and greedy
decoding.

## Running

```bash
# All pages with reference text, 8 parallel requests, Qwen3.5-9B on :8010, report GPU1 only
uv run ocrbench run --gpus 1

# Nanonets-OCR2 server with its own prompt
uv run ocrbench run --base-url http://localhost:8003/v1 --prompt nanonets --gpus 0

# Concurrency sweep -> results/<name>/scaling.csv
uv run ocrbench run --concurrency 1,4,8,16 --name sweep

# Big PDFs: every page of every matched document (timing/memory where no reference)
uv run ocrbench run --scope all --concurrency 16 --gpus 1

# Another server or model
uv run ocrbench run --base-url http://host:8000/v1 --model my-ocr --prompt-file prompt.txt

# Re-score saved predictions after changing normalization (no OCR calls)
uv run ocrbench score results/sweep
```

Useful options: `--only <name-part>`, `--max-pages-per-doc N`, `--limit N`,
`--prompt generic|nanonets|lightonocr|surya|deepseek-ocr`, `--prompt-file`, `--extra-body`, `--no-stream`, `--dpi`, `--max-side`, `--image-format jpeg`, `--max-tokens`, `--lowercase`,
`--render-workers`, `--prefetch`, `--warmup`. See `uv run ocrbench run -h`.

Pages are rendered one at a time in a process pool and queued ahead of the OCR
workers by at most `--prefetch` pages. Client memory therefore depends on
concurrency, not on PDF length.

## Output (`results/<run>/`)

- `pages.jsonl`: one line per page, written as each page finishes. It holds the prediction, timings, tokens, image size, RSS and CER/WER.
- `documents.csv`: per-document CER/WER, latency and errors.
- `summary.json`: all aggregate metrics.
- `run_meta.json`: config, wall time and resource samples.
- `scaling.csv`: sweep runs only.

## Normalization

Both sides go through the same rules before CER/WER:

- Unicode NFC.
- `[HANDWRITTEN: x]` becomes `x`.
- Other annotation tags such as `[EMBLEM]`, `[STAMP: ...]`, `<img>…</img>`, `<signature>…</signature>` and Markdown images `![…](…)` are dropped.
- LaTeX text wrappers are unwrapped: `$\text{2026 йил}$` becomes `2026 йил`.
- HTML tags are stripped and Markdown emphasis, heading and table syntax removed.
- Apostrophe variants (`‘ ’ ʻ ʼ`), quotes (`« » “ ”`) and dashes are unified.
- Whitespace is collapsed.
