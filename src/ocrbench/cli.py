"""Command-line entry point: ``ocrbench {list,prune,run,score}``."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from . import report
from .client import PROMPTS, OCRClient
from .gt import load_ground_truth, match_documents
from .pdf import RenderOptions, page_count
from .runner import RunConfig, build_jobs, rescore, run, summarize
from .server_metrics import VLLMMetrics


def _matches(args: argparse.Namespace):
    gt = load_ground_truth(args.gt)
    return gt, match_documents(gt, args.docs_dir)


def cmd_list(args: argparse.Namespace) -> int:
    _, m = _matches(args)
    print(f"{'file':60s} {'status':32s} {'gt pages':>8s} {'ref pages':>9s}" + ("  pdf pages" if args.check_pages else ""))
    for d in m.matched:
        line = f"{d.path.name[:60]:60s} {d.gt.status[:32]:32s} {d.gt.page_count:8d} {len(d.gt.pages) if d.gt.has_reference else 0:9d}"
        if args.check_pages:
            actual = page_count(str(d.path))
            line += f"  {actual:9d}" + ("" if actual == d.gt.page_count else "  MISMATCH")
        print(line)
    print(f"\nmatched {len(m.matched)}  with reference text {sum(d.gt.has_reference for d in m.matched)}")
    if m.without_gt:
        print(f"\nfiles without ground truth ({len(m.without_gt)}; never opened, remove with `ocrbench prune --yes`):")
        for p in m.without_gt:
            print(f"  {p.name}")
    if m.missing_files:
        print(f"\nground-truth entries without a file ({len(m.missing_files)}):")
        for d in m.missing_files:
            print(f"  {d.name}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    _, m = _matches(args)
    if not m.without_gt:
        print("every file has a ground-truth entry; nothing to remove")
        return 0
    for p in m.without_gt:
        if args.yes:
            p.unlink()
            print(f"removed {p.name}")
        else:
            print(f"would remove {p.name}")
    if not args.yes:
        print(f"\n{len(m.without_gt)} file(s) without ground truth; rerun with --yes to delete them")
    return 0


async def _run_all(args: argparse.Namespace) -> int:
    _, m = _matches(args)
    jobs = build_jobs(
        m.matched, scope=args.scope, name_filter=args.only, max_pages_per_doc=args.max_pages_per_doc
    )
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print("no pages selected", file=sys.stderr)
        return 1
    print(
        f"{len(jobs)} pages from {len({j.doc.path for j in jobs})} documents "
        f"({sum(j.reference is not None for j in jobs)} with reference text)",
        file=sys.stderr,
    )
    prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else PROMPTS[args.prompt]
    extra_body = json.loads(args.extra_body) if args.extra_body else {}
    if not args.thinking:
        # Qwen3/3.5 chat templates think by default; OCR does not need it.
        extra_body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    gpus = tuple(int(g) for g in args.gpus.split(",")) if args.gpus else None
    levels = sorted({int(c) for c in args.concurrency.split(",")})
    root = Path(args.out) / (args.name or datetime.now().strftime("%Y%m%d-%H%M%S"))
    render = RenderOptions(args.dpi, args.max_side, args.image_format, args.jpeg_quality)
    summaries = []
    for c in levels:
        out_dir = root / f"c{c}" if len(levels) > 1 else root
        cfg = RunConfig(
            concurrency=c,
            render_workers=args.render_workers,
            prefetch=args.prefetch or max(4, c),
            render=render,
            warmup=args.warmup,
            lowercase=args.lowercase,
            gpu_monitor=not args.no_gpu,
            gpus=gpus,
            quiet=args.quiet,
        )
        async with OCRClient(
            args.base_url,
            args.model,
            api_key=args.api_key or os.environ.get("OPENAI_API_KEY"),
            prompt=prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout,
            max_retries=args.retries,
            max_connections=c,
            stream=not args.no_stream,
            extra_body=extra_body,
        ) as client:
            print(f"\nrunning {client.model} at {args.base_url}, concurrency {c}", file=sys.stderr)
            server = None if args.no_server_metrics else VLLMMetrics(args.base_url)
            meta = await run(jobs, client, cfg, out_dir, server)
        summary = summarize(_load_records(out_dir), meta)
        report.write(summary, out_dir)
        report.print_summary(summary, out_dir)
        summaries.append(summary)
    if len(summaries) > 1:
        report.write_scaling(summaries, root)
    return 0


def _load_records(out_dir: Path) -> list[dict]:
    records = []
    with open(out_dir / "pages.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec.pop("prediction", None)
            records.append(rec)
    return records


def cmd_score(args: argparse.Namespace) -> int:
    """Re-score saved predictions (e.g. after changing normalization) without re-running OCR."""
    gt = load_ground_truth(args.gt)
    by_name = {d.name: d for d in gt.values()}
    run_dirs = sorted(p.parent for p in Path(args.run_dir).glob("**/pages.jsonl"))
    if not run_dirs:
        print(f"no pages.jsonl under {args.run_dir}", file=sys.stderr)
        return 1
    for out_dir in run_dirs:
        meta = json.loads((out_dir / "run_meta.json").read_text())
        summary = summarize(rescore(out_dir / "pages.jsonl", by_name, args.lowercase), meta)
        report.write(summary, out_dir)
        report.print_summary(summary, out_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ocrbench", description="OCR benchmark: CER/WER, latency, throughput, memory.")
    parser.add_argument("--gt", default="ground_truth.json", help="ground-truth JSON (default: %(default)s)")
    parser.add_argument("--docs-dir", default="test_uchun_ddoklar", help="PDF folder (default: %(default)s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="show which files match the ground truth")
    p.add_argument("--check-pages", action="store_true", help="open matched PDFs and compare page counts")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("prune", help="delete files that have no ground-truth entry")
    p.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("run", help="run the benchmark against an OpenAI-compatible endpoint")
    p.add_argument("--base-url", default="http://localhost:8010/v1", help="default: %(default)s (scripts/serve_qwen35.sh)")
    p.add_argument("--model", default=None, help="model id (default: first model the server lists)")
    p.add_argument("--api-key", default=None, help="default: $OPENAI_API_KEY")
    p.add_argument("--concurrency", default="8", help="parallel requests; comma list runs a sweep, e.g. 1,4,16")
    p.add_argument("--scope", choices=("scored", "all"), default="scored",
                   help="scored: pages with reference text; all: every page of every matched PDF (timing only where no reference)")
    p.add_argument("--only", nargs="+", help="only documents whose name contains one of these strings")
    p.add_argument("--max-pages-per-doc", type=int)
    p.add_argument("--limit", type=int, help="stop after this many pages in total")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--max-side", type=int, default=1800, help="cap on the longest image side in pixels (0 = none)")
    p.add_argument("--image-format", choices=("png", "jpeg"), default="png")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--prompt", choices=sorted(PROMPTS), default="generic",
                   help="built-in prompt: generic (Qwen-VL etc.) or a model's own prompt "
                        "(lightonocr sends the image only) (default: %(default)s)")
    p.add_argument("--prompt-file", help="file with the OCR prompt (overrides --prompt)")
    p.add_argument("--thinking", action="store_true", help="leave reasoning mode on (default: enable_thinking=false)")
    p.add_argument("--extra-body", help='JSON merged into each request, e.g. \'{"top_k": 1}\'')
    p.add_argument("--no-stream", action="store_true", help="non-streaming requests (no TTFT)")
    p.add_argument("--no-server-metrics", action="store_true", help="do not scrape the vLLM /metrics endpoint")
    p.add_argument("--gpus", help="GPU indices to report, e.g. 1 (default: all)")
    p.add_argument("--timeout", type=float, default=600.0, help="per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--render-workers", type=int, default=4)
    p.add_argument("--prefetch", type=int, help="rendered pages queued ahead of OCR (default: max(4, concurrency))")
    p.add_argument("--warmup", type=int, default=1, help="untimed requests before measuring")
    p.add_argument("--lowercase", action="store_true", help="case-insensitive CER/WER")
    p.add_argument("--no-gpu", action="store_true", help="skip nvidia-smi sampling")
    p.add_argument("--out", default="results")
    p.add_argument("--name", help="run directory name (default: timestamp)")
    p.add_argument("-q", "--quiet", action="store_true")
    p.set_defaults(func=lambda a: asyncio.run(_run_all(a)))

    p = sub.add_parser("score", help="re-score a finished run from its saved predictions")
    p.add_argument("run_dir")
    p.add_argument("--lowercase", action="store_true")
    p.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)
