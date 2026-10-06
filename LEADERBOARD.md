# OCR leaderboard

73 pages from 13 documents in `test_uchun_ddoklar/` (123,533 reference
characters), scored against `ground_truth.json`. Each model was served with
vLLM on one NVIDIA L40 and given the same page images (150 DPI, longest side
1800 px) with greedy decoding. Runs date from 2026-10-05.

Models are ranked by CER. Accuracy comes from the concurrency-1 run. The best
value in each column is in bold.

## Columns

| Column | Meaning |
|---|---|
| **CER** | Character error rate: wrong, missing and extra characters ÷ all reference characters, over all 73 pages. Lower is better. |
| **WER** | Word error rate: the same as CER, but counted in words. |
| **Page CER p50 / p90** | CER for each page separately. p50 is a typical page (the median). p90 is a bad page: 90% of pages do better than this. |
| **Truncated pages** | Pages where the model hit its output token limit, usually because it kept repeating text. |
| **Latency p50 (c1)** | Median time from sending a page to receiving its full text, with one page sent at a time. |
| **Pages/s (c1)** | Pages finished per second with one request at a time. 0.19 pages/s means one page every 1 ÷ 0.19 ≈ 5.2 s on average. |
| **Pages/s (c16)** | Pages finished per second with 16 requests at once. The GPU batches them, so throughput is higher. |

CER and WER are measured after [normalization](README.md#normalization), so
Markdown and HTML markup are not counted as errors.

## Leaderboard

| # | Model | CER | WER | Page CER p50 / p90 | Truncated pages | Latency p50 (c1) | Pages/s (c1) | Pages/s (c16) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | [Surya OCR 2](https://huggingface.co/datalab-to/surya-ocr-2) | **0.72%** | 2.47% | 0.17% / 1.92% | 1 | 5.7 s | 0.19 | 1.48 |
| 2 | [LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1.04% | **2.38%** | 0.13% / 3.02% | 0 | 3.5 s | 0.31 | 1.13 |
| 3 | [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 2.43% | 4.49% | 0.37% / 5.29% | 1 | 20.1 s | 0.05 | 0.42 |
| 4 | [dots.ocr](https://huggingface.co/dots-studio/dots.ocr) | 3.04% | 3.98% | 0.24% / 10.72% | 0 | 6.7 s | 0.17 | 0.86 |
| 5 | Nanonets-OCR2-3B | 3.54% | 17.74% | 3.42% / 5.72% | 0 | 12.2 s | 0.09 | 0.59 |
| 6 | [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | 4.56% | 23.05% | 2.80% / 11.58% | 1 | 2.9 s | 0.34 | 1.35 |
| 7 | [Qwen2.5-VL-3B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 7.94% | 23.60% | 4.12% / 25.73% | 1 | 10.0 s | 0.11 | 0.83 |
| 8 | [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 15.06% | 49.24% | 5.21% / 40.98% | 5 | **1.9 s** | **0.54** | **3.16** |
| 9 | [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 15.98% | 44.51% | 10.36% / 25.79% | 2 | 2.7 s | 0.38 | 1.40 |
| 10 | [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | 20.19% | 26.60% | 0.68% / 18.41% | 3 | 2.8 s | 0.34 | 1.42 |
| 11 | [Qwen3-VL-4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | 103.98% | 121.58% | 25.65% / 371.48% | 17 | 12.6 s | 0.05 | 0.32 |
| 12 | [MinerU2.5-1.2B](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | 143.08% | 137.37% | 92.64% / 318.21% | 29 | 3.4 s | 0.19 | 1.32 |

## Takeaways

- **Most accurate:** Surya OCR 2 has the lowest CER (0.72%). LightOnOCR-2-1B has the lowest WER (2.38%) and is close on CER.
- **Accurate and fast:** Among the models under 5% CER, HunyuanOCR is the fastest per request (2.9 s per page), but its WER is 23%. LightOnOCR-2-1B is next (3.5 s) with 2.4% WER. Under load, Surya OCR 2 has the highest throughput of this group (1.48 pages/s at c16).
- **Fastest:** PaddleOCR-VL-1.5 reaches 3.16 pages/s at c16, but its CER is 15%.
- Qwen3.5-9B ranks third on accuracy, but it is the slowest model at 20 s per page.

## Caveats

- **CER above 100%.** Qwen3-VL-4B and MinerU2.5 fall into repetition loops. Their output contains about 100k and 95k inserted characters, and 17 and 29 of 73 pages hit the token limit. A short page that repeats until the limit can have a page CER well above 100%.
- **DeepSeek-OCR-2** has a median page CER of 0.68%, but a few pages that loop (21k inserted characters) raise its overall CER to 20%.
- **Layout-pipeline models.** PaddleOCR-VL-1.5, MinerU2.5 and GLM-OCR are normally given layout crops. Here they get the whole page, which works against them.
- **dots.ocr** skips page headers and footers by design. Most of its errors are deletions.
- **Token limit.** Surya OCR 2 had an 8192-token output limit. Every other model had 4096.
- **Nanonets-OCR2-3B** was served separately, on GPU 0 with `gpu_memory_utilization` 0.4. The other models ran on GPU 1 with 0.6.
- **Run-to-run variation.** CER changes between concurrency levels for DeepSeek-OCR-2 (15.4–27.6%), PaddleOCR-VL-1.5 (15.1–26.3%) and GLM-OCR (12.5–20.8%), because batched decoding is not bit-exact. The top six models stay within 0.3 points.
- The ground truth is transcribed but has not been proofread (`transcribed_unproofread`), so differences under about 0.5% CER are not meaningful.

## Reproducing

```bash
scripts/bench_models.sh                     # serve each model, run the 1/4/8/16 sweep
uv run python scripts/compare_models.py     # -> results/comparison.csv
```

Per-run details are in `results/<model>_sweep/c<N>/summary.json`.
