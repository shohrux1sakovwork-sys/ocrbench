"""Compare concurrency sweeps of several models: results/*_sweep -> comparison table.

Accuracy comes from the concurrency-1 run; speed columns from c1 and the
highest concurrency level of each sweep.

    uv run python scripts/compare_models.py [results_dir]
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def load(sweep: Path) -> dict[int, dict]:
    runs = {}
    for summary in sweep.glob("c*/summary.json"):
        runs[int(summary.parent.name[1:])] = json.loads(summary.read_text(encoding="utf-8"))
    return runs


def row(runs: dict[int, dict]) -> dict:
    lo, hi = runs[min(runs)], runs[max(runs)]
    acc, t1, th = lo["accuracy"], lo["time"], hi["time"]
    dist = acc["cer_page_distribution"]
    return {
        "model": lo["model_id"],
        "cer": acc["cer"],
        "wer": acc["wer"],
        "page_cer_p50": dist["p50"],
        "page_cer_p90": dist["p90"],
        "truncated": max(r["truncated_max_tokens"] for r in runs.values()),
        "errors": max(r["errors"] for r in runs.values()),
        "c1_latency_p50_s": t1["latency_s"]["p50"],
        "c1_ttft_p50_s": (t1.get("ttft_s") or {}).get("p50"),
        "c1_decode_tok_s": (t1.get("decode_tokens_per_s_per_request") or {}).get("p50"),
        "c1_pages_per_s": t1["pages_per_s"],
        "max_concurrency": max(runs),
        "max_pages_per_s": th["pages_per_s"],
        "speedup": th["pages_per_s"] / t1["pages_per_s"],
        "cer_range": (
            min(r["accuracy"]["cer"] for r in runs.values()),
            max(r["accuracy"]["cer"] for r in runs.values()),
        ),
    }


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows = [row(runs) for sweep in sorted(root.glob("*_sweep")) if (runs := load(sweep))]
    rows.sort(key=lambda r: r["cer"])

    out = root / "comparison.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow(r | {"cer_range": "%.4f-%.4f" % r["cer_range"]})

    print(
        "| Model | CER | WER | page CER p50 / p90 | CER across levels | truncated | "
        "c1 latency p50 | c1 decode tok/s | c1 pages/s | c16 pages/s | speedup |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['model']} | {r['cer']:.2%} | {r['wer']:.2%} "
            f"| {r['page_cer_p50']:.2%} / {r['page_cer_p90']:.2%} "
            f"| {r['cer_range'][0]:.2%}–{r['cer_range'][1]:.2%} | {r['truncated']} "
            f"| {r['c1_latency_p50_s']:.1f}s | {r['c1_decode_tok_s'] or 0:.0f} "
            f"| {r['c1_pages_per_s']:.3f} | {r['max_pages_per_s']:.3f} | {r['speedup']:.1f}× |"
        )
    print(f"\n{out}")


if __name__ == "__main__":
    main()
