"""Writing and printing benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v * 100:.2f}%"


def _num(v: float | None, fmt: str = ".2f") -> str:
    return "-" if v is None else format(v, fmt)


def write(summary: dict, out_dir: Path) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    docs = summary["documents"]
    if docs:
        with open(out_dir / "documents.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(docs[0]))
            writer.writeheader()
            writer.writerows(docs)


def print_summary(s: dict, out_dir: Path) -> None:
    acc, t, sp, cx = s["accuracy"], s["time"], s["space"], s["complexity"]
    cfg = s["config"]
    print(f"\n=== {s['model_id']} | concurrency {cfg['concurrency']} | {out_dir} ===")
    print(
        f"pages {s['pages']}  ok {s['pages_ok']}  errors {s['errors']}  "
        f"truncated(max_tokens) {s['truncated_max_tokens']}  retried {s['retried']}"
    )

    print("\nAccuracy (normalized; raw = markup counted as errors)")
    if acc["scored_pages"]:
        print(
            f"  CER {_pct(acc['cer'])}  WER {_pct(acc['wer'])}  "
            f"(page mean CER {_pct(acc['cer_page_mean'])}  WER {_pct(acc['wer_page_mean'])})"
        )
        print(f"  raw CER {_pct(acc['raw_cer'])}  raw WER {_pct(acc['raw_wer'])}  scored pages {acc['scored_pages']}")
        ce = acc["char_edits"]
        print(f"  char edits: sub {ce['substitutions']}  ins {ce['insertions']}  del {ce['deletions']}  ref {ce['ref_len']}")
    else:
        print("  no pages with reference text")

    lat = t["latency_s"]
    print("\nTime")
    print(
        f"  wall {t['wall_s']:.1f}s  throughput {_num(t['pages_per_s'], '.3f')} pages/s  "
        f"{_num(t['completion_tokens_per_s'], '.0f')} output tok/s  parallelism {_num(t['effective_parallelism'])}"
    )
    if lat:
        print(
            f"  latency p50 {lat['p50']:.2f}s  p90 {lat['p90']:.2f}s  p99 {lat['p99']:.2f}s  "
            f"max {lat['max']:.2f}s  | render mean {t['render_s'].get('mean', 0) * 1000:.0f}ms"
        )
    if t.get("ttft_s"):
        print(
            f"  TTFT p50 {t['ttft_s']['p50']:.2f}s  p90 {t['ttft_s']['p90']:.2f}s  | "
            f"decode p50 {_num(t['decode_tokens_per_s_per_request'].get('p50'), '.1f')} tok/s per request"
        )

    m = s.get("model")
    if m:
        g, tot, h = m["gauges"], m["totals"], m["histograms_mean"]
        print("\nModel server (vLLM /metrics)")
        if "kv_cache_usage" in g:
            print(
                f"  KV cache usage peak {g['kv_cache_usage']['peak'] * 100:.1f}%  mean {g['kv_cache_usage']['mean'] * 100:.1f}%  "
                f"| running peak {g.get('requests_running', {}).get('peak', 0):.0f}  "
                f"waiting peak {g.get('requests_waiting', {}).get('peak', 0):.0f}  preemptions {_num(tot['preemptions'], '.0f')}"
            )
        print(
            "  mean per request: "
            + "  ".join(f"{k.removesuffix('_s')} {_num(v, '.3f')}s" for k, v in h.items() if v is not None)
        )
        print(
            f"  tokens: prompt {_num(tot['prompt_tokens'], '.0f')}  generated {_num(tot['generation_tokens'], '.0f')}  "
            f"prefix-cache hit rate {_pct(m.get('prefix_cache_hit_rate'))}"
        )
        if m.get("cache_config"):
            print("  cache config: " + "  ".join(f"{k}={v}" for k, v in m["cache_config"].items()))
        if (m.get("prefix_cache_hit_rate") or 0) > 0.2:
            print(
                "  WARNING: high prefix-cache hit rate; repeated pages skip prefill, so TTFT and "
                "throughput are optimistic. Serve with --no-enable-prefix-caching for model timing."
            )

    print("\nEmpirical complexity (cost ≈ c·size^k)")
    for name, fit in cx.items():
        if fit:
            print(f"  {name:28s} k={fit['exponent']:+.2f}  r²={fit['r2']:.2f}  {fit['order']}  (n={fit['n']})")

    print("\nSpace")
    print(
        f"  client RSS peak {sp['rss_peak_mb']:.0f} MB (baseline {sp['rss_baseline_mb']:.0f} MB)  "
        f"growth {_num(sp['rss_growth_mb_per_page'], '+.3f')} MB/page  "
        f"in-flight images peak {sp['peak_inflight_image_mb']:.1f} MB"
    )
    if sp["image_kb"]:
        print(
            f"  page image mean {sp['image_kb']['mean']:.0f} KB, {sp['image_pixels']['mean'] / 1e6:.2f} MP  "
            f"prompt tokens mean {_num(sp['prompt_tokens'].get('mean'), '.0f')}"
        )
    for idx, g in (sp.get("gpus") or {}).items():
        print(
            f"  GPU{idx} mem peak {g['memory_peak_mib']:.0f} MiB (baseline {g['memory_baseline_mib']:.0f})  "
            f"util mean {g['utilization_mean_pct']:.0f}% (baseline {g['utilization_baseline_pct']:.0f}%)"
        )

    if s["documents"]:
        print("\nPer document")
        print(f"  {'file':48s} {'pages':>5s} {'err':>3s} {'CER':>8s} {'WER':>8s} {'lat/pg':>7s}")
        for d in s["documents"]:
            print(
                f"  {d['file'][:48]:48s} {d['pages']:5d} {d['errors']:3d} {_pct(d.get('cer')):>8s} "
                f"{_pct(d.get('wer')):>8s} {_num(d['latency_mean_s']):>6s}s"
            )


def write_scaling(summaries: list[dict], out_dir: Path) -> None:
    """Throughput at each concurrency level relative to the lowest level."""
    base = summaries[0]
    base_tp, base_c = base["time"]["pages_per_s"], base["config"]["concurrency"]
    rows = []
    for s in summaries:
        c, tp = s["config"]["concurrency"], s["time"]["pages_per_s"]
        speedup = tp / base_tp if tp and base_tp else None
        rows.append(
            {
                "concurrency": c,
                "pages_per_s": tp,
                "speedup": speedup,
                "efficiency": speedup / (c / base_c) if speedup else None,
                "latency_p50_s": s["time"]["latency_s"].get("p50"),
                "latency_p99_s": s["time"]["latency_s"].get("p99"),
                "cer": s["accuracy"].get("cer"),
                "wer": s["accuracy"].get("wer"),
                "errors": s["errors"],
                "rss_peak_mb": s["space"]["rss_peak_mb"],
            }
        )
    with open(out_dir / "scaling.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\n=== Concurrency scaling ===")
    print(f"  {'conc':>4s} {'pages/s':>8s} {'speedup':>7s} {'effic.':>6s} {'p50':>7s} {'p99':>7s} {'CER':>8s} {'err':>3s}")
    for r in rows:
        print(
            f"  {r['concurrency']:4d} {_num(r['pages_per_s'], '.3f'):>8s} {_num(r['speedup']):>7s} "
            f"{_pct(r['efficiency']):>6s} {_num(r['latency_p50_s']):>6s}s {_num(r['latency_p99_s']):>6s}s "
            f"{_pct(r['cer']):>8s} {r['errors']:3d}"
        )
