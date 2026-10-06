"""Streaming benchmark pipeline: render (process pool) -> OCR (async workers) -> score.

Rendering runs ahead of OCR by at most ``prefetch`` pages, so the number of
page images held in memory is bounded by render workers + prefetch +
concurrency regardless of how many pages a PDF has.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from .client import OCRClient
from .gt import GTDocument, MatchedDocument
from .metrics import EditCounts, complexity_fit, linear_fit, percentiles, score_page
from .monitor import ResourceMonitor
from .pdf import RenderOptions, render_page
from .server_metrics import VLLMMetrics


@dataclass(frozen=True)
class PageJob:
    doc: MatchedDocument
    page: int

    @property
    def reference(self) -> str | None:
        return self.doc.gt.reference(self.page)


def build_jobs(
    docs: list[MatchedDocument],
    *,
    scope: str,
    name_filter: list[str] | None = None,
    max_pages_per_doc: int | None = None,
) -> list[PageJob]:
    """``scope="scored"``: pages with reference text. ``"all"``: every page of
    every matched document; pages without a reference are timed but not scored."""
    jobs = []
    for doc in docs:
        if name_filter and not any(f in doc.path.name or f in doc.gt.name for f in name_filter):
            continue
        if scope == "scored":
            pages = sorted(doc.gt.pages) if doc.gt.has_reference else []
        else:
            pages = list(range(1, doc.gt.page_count + 1))
        if max_pages_per_doc:
            pages = pages[:max_pages_per_doc]
        jobs.extend(PageJob(doc, p) for p in pages)
    return jobs


@dataclass(frozen=True)
class RunConfig:
    concurrency: int
    render_workers: int
    prefetch: int
    render: RenderOptions
    warmup: int = 1
    lowercase: bool = False
    gpu_monitor: bool = True
    gpus: tuple[int, ...] | None = None
    quiet: bool = False


def _render_pool(workers: int) -> ProcessPoolExecutor:
    # "spawn" avoids forking a process that already runs threads and an event loop.
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


class _Progress:
    def __init__(self, total: int, quiet: bool) -> None:
        self.total, self.quiet, self.done = total, quiet, 0
        self.tty = sys.stderr.isatty()
        self.start = time.perf_counter()

    def update(self, rec: dict) -> None:
        self.done += 1
        if self.quiet or (not self.tty and self.done % 10 and self.done != self.total):
            return
        elapsed = time.perf_counter() - self.start
        status = (
            "ERROR"
            if rec.get("error")
            else f"{rec['latency_s']:.2f}s" + (f" cer={rec['cer']:.3f}" if rec.get("cer") is not None else "")
        )
        line = (
            f"[{self.done}/{self.total}] {self.done / elapsed:.2f} pages/s | "
            f"{rec['file'][:40]} p{rec['page']} {status}"
        )
        if self.tty:
            print(f"\r\033[K{line}", end="" if self.done < self.total else "\n", file=sys.stderr, flush=True)
        else:
            print(line, file=sys.stderr, flush=True)


async def run(
    jobs: list[PageJob], client: OCRClient, cfg: RunConfig, out_dir: Path, server: VLLMMetrics | None = None
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    monitor = ResourceMonitor(gpu=cfg.gpu_monitor, gpus=cfg.gpus)
    queue: asyncio.Queue = asyncio.Queue(maxsize=cfg.prefetch)
    progress = _Progress(len(jobs), cfg.quiet)
    inflight = {"bytes": 0, "peak": 0}

    with _render_pool(cfg.render_workers) as pool, open(out_dir / "pages.jsonl", "w", encoding="utf-8") as sink:
        if jobs and cfg.warmup:
            # Untimed requests so model/server warm-up does not skew latency.
            first = await loop.run_in_executor(pool, render_page, str(jobs[0].doc.path), jobs[0].page, cfg.render)
            for _ in range(cfg.warmup):
                await client.ocr(first["image_url"])
            del first

        async def produce() -> None:
            pending: deque = deque()

            async def hand_over() -> None:
                job, future = pending.popleft()
                try:
                    rendered, error = await future, None
                except Exception as exc:  # noqa: BLE001 - recorded per page
                    rendered, error = None, f"render: {type(exc).__name__}: {exc}"
                if rendered:
                    inflight["bytes"] += rendered["image_bytes"]
                    inflight["peak"] = max(inflight["peak"], inflight["bytes"])
                await queue.put((job, rendered, error))

            try:
                for job in jobs:
                    future = loop.run_in_executor(pool, render_page, str(job.doc.path), job.page, cfg.render)
                    pending.append((job, future))
                    if len(pending) >= cfg.render_workers * 2:
                        await hand_over()
                while pending:
                    await hand_over()
            finally:
                for _ in range(cfg.concurrency):
                    await queue.put(None)

        async def consume() -> None:
            while (item := await queue.get()) is not None:
                job, rendered, error = item
                rec = await _process(job, rendered, error, client, cfg)
                if rendered:
                    inflight["bytes"] -= rendered["image_bytes"]
                del rendered, item
                rec["completed_at_s"] = time.perf_counter() - started
                rec["rss_mb"] = monitor.rss_mb()
                sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
                sink.flush()
                progress.update(rec)

        if server:
            await server.start()
        monitor.start()
        started = time.perf_counter()
        progress.start = started
        await asyncio.gather(produce(), *(consume() for _ in range(cfg.concurrency)))
        wall_s = time.perf_counter() - started
        resources = monitor.stop()
        server_stats = await server.stop() if server else None

    resources["peak_inflight_image_mb"] = inflight["peak"] / (1024 * 1024)
    meta = {
        "wall_s": wall_s,
        "model": client.model,
        "config": {**asdict(cfg), "max_tokens": client.max_tokens, "prompt": client.prompt},
        "resources": resources,
        "server": server_stats,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


async def _process(job: PageJob, rendered: dict | None, error: str | None, client: OCRClient, cfg: RunConfig) -> dict:
    rec: dict = {
        "file": job.doc.gt.name,
        "path": job.doc.path.name,
        "page": job.page,
        "has_reference": job.reference is not None,
        "error": error,
    }
    if rendered is None:
        return rec
    rec.update(
        width=rendered["width"],
        height=rendered["height"],
        pixels=rendered["width"] * rendered["height"],
        image_bytes=rendered["image_bytes"],
        render_s=rendered["render_s"],
    )
    try:
        response = await client.ocr(rendered["image_url"])
    except Exception as exc:  # noqa: BLE001 - recorded per page
        rec["error"] = f"ocr: {type(exc).__name__}: {exc}"
        return rec
    rec.update(
        latency_s=response.latency_s,
        ttft_s=response.ttft_s,
        attempts=response.attempts,
        prompt_tokens=response.prompt_tokens,
        completion_tokens=response.completion_tokens,
        finish_reason=response.finish_reason,
        pred_chars=len(response.text),
        prediction=response.text,
    )
    if job.reference is not None:
        rec.update(score_page(job.reference, response.text, lowercase=cfg.lowercase))
    return rec


def rescore(pages_path: Path, gt_by_name: dict[str, GTDocument], lowercase: bool) -> list[dict]:
    """Recompute accuracy for a finished run from its saved predictions."""
    records = []
    with open(pages_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            for key in ("cer", "wer", "raw_cer", "raw_wer", "chars", "words", "raw_chars", "raw_words"):
                rec.pop(key, None)
            doc = gt_by_name.get(rec["file"])
            reference = doc.reference(rec["page"]) if doc else None
            rec["has_reference"] = reference is not None
            if reference is not None and rec.get("prediction") is not None:
                rec.update(score_page(reference, rec["prediction"], lowercase=lowercase))
            rec.pop("prediction", None)
            records.append(rec)
    return records


def _accuracy(recs: list[dict]) -> dict:
    scored = [r for r in recs if r.get("cer") is not None]
    if not scored:
        return {"scored_pages": 0}
    totals = {k: EditCounts() for k in ("chars", "words", "raw_chars", "raw_words")}
    for r in scored:
        for k, acc in totals.items():
            acc += EditCounts.from_dict(r[k])
    return {
        "scored_pages": len(scored),
        # Micro: total edits / total reference length (long pages weigh more).
        "cer": totals["chars"].rate,
        "wer": totals["words"].rate,
        "raw_cer": totals["raw_chars"].rate,
        "raw_wer": totals["raw_words"].rate,
        # Macro: mean of per-page rates (every page weighs the same).
        "cer_page_mean": sum(r["cer"] for r in scored) / len(scored),
        "wer_page_mean": sum(r["wer"] for r in scored) / len(scored),
        "cer_page_distribution": percentiles([r["cer"] for r in scored]),
        "char_edits": asdict(totals["chars"]),
        "word_edits": asdict(totals["words"]),
    }


def summarize(records: list[dict], meta: dict) -> dict:
    ok = [r for r in records if r.get("latency_s") is not None]
    wall = meta["wall_s"]
    latencies = [r["latency_s"] for r in ok]
    completion = [r.get("completion_tokens") or 0 for r in ok]

    per_doc: dict[str, list[dict]] = {}
    for r in records:
        per_doc.setdefault(r["file"], []).append(r)
    documents = []
    for name, recs in per_doc.items():
        doc_ok = [r for r in recs if r.get("latency_s") is not None]
        documents.append(
            {
                "file": name,
                "pages": len(recs),
                "errors": sum(1 for r in recs if r.get("error")),
                "truncated": sum(1 for r in recs if r.get("finish_reason") == "length"),
                "latency_mean_s": sum(r["latency_s"] for r in doc_ok) / len(doc_ok) if doc_ok else None,
                "latency_sum_s": sum(r["latency_s"] for r in doc_ok),
                **{
                    k: v
                    for k, v in _accuracy(recs).items()
                    if k in ("scored_pages", "cer", "wer", "raw_cer", "raw_wer", "cer_page_mean")
                },
            }
        )

    # Space: with streaming, RSS should stay flat as pages accumulate, i.e. a
    # slope near zero MB/page means memory is O(1) in document length.
    mem_points = [(i + 1, r["rss_mb"]) for i, r in enumerate(sorted(records, key=lambda r: r["completed_at_s"]))]
    mem_fit = linear_fit([p[0] for p in mem_points], [p[1] for p in mem_points])

    by_doc_size = [
        (len(recs), d["latency_sum_s"]) for d, recs in ((d, per_doc[d["file"]]) for d in documents) if d["latency_sum_s"]
    ]
    return {
        "model_id": meta["model"],
        "pages": len(records),
        "pages_ok": len(ok),
        "errors": sum(1 for r in records if r.get("error")),
        "truncated_max_tokens": sum(1 for r in records if r.get("finish_reason") == "length"),
        "retried": sum(1 for r in ok if (r.get("attempts") or 1) > 1),
        "accuracy": _accuracy(records),
        "time": {
            "wall_s": wall,
            "pages_per_s": len(ok) / wall if wall else None,
            "s_per_page_effective": wall / len(ok) if ok else None,
            "completion_tokens_per_s": sum(completion) / wall if wall else None,
            # Average number of requests in flight; ≈ concurrency when the
            # client keeps the server saturated.
            "effective_parallelism": sum(latencies) / wall if wall else None,
            "latency_s": percentiles(latencies),
            "render_s": percentiles([r["render_s"] for r in records if r.get("render_s") is not None]),
            # Client-measured; TTFT includes queueing at the server plus prefill.
            "ttft_s": percentiles([r["ttft_s"] for r in ok if r.get("ttft_s") is not None]),
            "decode_tokens_per_s_per_request": percentiles(
                [
                    (r["completion_tokens"] - 1) / (r["latency_s"] - r["ttft_s"])
                    if r.get("ttft_s") is not None
                    else r["completion_tokens"] / r["latency_s"]
                    for r in ok
                    if (r.get("completion_tokens") or 0) > 1 and r["latency_s"] > (r.get("ttft_s") or 0)
                ]
            ),
        },
        "model": meta.get("server"),
        "complexity": {
            "latency_vs_output_tokens": complexity_fit(completion, latencies),
            "latency_vs_output_chars": complexity_fit([r["pred_chars"] for r in ok], latencies),
            "latency_vs_image_pixels": complexity_fit([r["pixels"] for r in ok], latencies),
            "ttft_vs_prompt_tokens": complexity_fit(
                [r["prompt_tokens"] for r in ok if r.get("ttft_s") and r.get("prompt_tokens")],
                [r["ttft_s"] for r in ok if r.get("ttft_s") and r.get("prompt_tokens")],
            ),
            "render_vs_image_pixels": complexity_fit(
                [r["pixels"] for r in records if r.get("render_s")], [r["render_s"] for r in records if r.get("render_s")]
            ),
            "doc_time_vs_doc_pages": complexity_fit([p[0] for p in by_doc_size], [p[1] for p in by_doc_size]),
        },
        "space": {
            **meta["resources"],
            "rss_growth_mb_per_page": mem_fit["slope"] if mem_fit else None,
            "image_kb": percentiles([r["image_bytes"] / 1024 for r in records if r.get("image_bytes")]),
            "image_pixels": percentiles([r["pixels"] for r in records if r.get("pixels")]),
            "prompt_tokens": percentiles([r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]),
        },
        "documents": documents,
        "config": meta["config"],
    }
