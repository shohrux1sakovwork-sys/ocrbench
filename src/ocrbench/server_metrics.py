"""Model-side measurements from a vLLM server's Prometheus ``/metrics`` endpoint.

Gauges (KV-cache usage, running/waiting requests) are sampled during the run;
counters and histograms are read before and after, and the difference covers
exactly the requests of this run (provided nothing else uses the server).
"""

from __future__ import annotations

import asyncio
import re
from collections import defaultdict

import httpx

_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")
_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

# Metric names changed across vLLM versions; the first one present is used.
_GAUGES = {
    "kv_cache_usage": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "requests_running": ("vllm:num_requests_running",),
    "requests_waiting": ("vllm:num_requests_waiting",),
}
_COUNTERS = {
    "prompt_tokens": ("vllm:prompt_tokens_total",),
    "generation_tokens": ("vllm:generation_tokens_total",),
    "preemptions": ("vllm:num_preemptions_total",),
    "prefix_cache_queries": ("vllm:prefix_cache_queries_total",),
    "prefix_cache_hits": ("vllm:prefix_cache_hits_total",),
}
_HISTOGRAMS = {
    "ttft_s": ("vllm:time_to_first_token_seconds",),
    "inter_token_latency_s": ("vllm:inter_token_latency_seconds", "vllm:time_per_output_token_seconds"),
    "e2e_latency_s": ("vllm:e2e_request_latency_seconds",),
    "queue_time_s": ("vllm:request_queue_time_seconds",),
    "prefill_time_s": ("vllm:request_prefill_time_seconds",),
    "decode_time_s": ("vllm:request_decode_time_seconds",),
}


def parse(text: str) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    """Return (metric name -> value summed over label sets, name -> labels of the
    last sample). Histogram buckets are skipped; only ``_sum``/``_count`` are kept."""
    values: dict[str, float] = defaultdict(float)
    labels: dict[str, dict[str, str]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m or m.group(1).endswith("_bucket"):
            continue
        try:
            values[m.group(1)] += float(m.group(3))
        except ValueError:
            continue
        if m.group(2):
            labels[m.group(1)] = dict(_LABEL.findall(m.group(2)))
    return dict(values), labels


def _first(values: dict[str, float], names: tuple[str, ...], suffix: str = "") -> float | None:
    for name in names:
        if name + suffix in values:
            return values[name + suffix]
    return None


class VLLMMetrics:
    def __init__(self, base_url: str, interval_s: float = 0.5) -> None:
        # The OpenAI API lives under /v1; metrics are served at the root.
        root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
        self._http = httpx.AsyncClient(base_url=root, timeout=10.0)
        self.interval_s = interval_s
        self.available = False
        self._before: dict[str, float] = {}
        self._labels: dict[str, dict[str, str]] = {}
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._task: asyncio.Task | None = None

    async def _fetch(self) -> tuple[dict[str, float], dict[str, dict[str, str]]] | None:
        try:
            response = await self._http.get("/metrics")
            response.raise_for_status()
        except httpx.HTTPError:
            return None
        return parse(response.text)

    async def start(self) -> None:
        fetched = await self._fetch()
        if fetched is None or not any(k.startswith("vllm:") for k in fetched[0]):
            await self._http.aclose()
            return
        self.available = True
        self._before, self._labels = fetched
        self._task = asyncio.create_task(self._sample())

    async def _sample(self) -> None:
        while True:
            fetched = await self._fetch()
            if fetched:
                for key, names in _GAUGES.items():
                    value = _first(fetched[0], names)
                    if value is not None:
                        self._samples[key].append(value)
            await asyncio.sleep(self.interval_s)

    async def stop(self) -> dict | None:
        if not self.available:
            return None
        self._task.cancel()
        after, _ = await self._fetch() or ({}, {})
        await self._http.aclose()

        def delta(names: tuple[str, ...], suffix: str = "") -> float | None:
            a, b = _first(after, names, suffix), _first(self._before, names, suffix)
            return None if a is None or b is None else a - b

        out: dict = {"gauges": {}, "totals": {}, "histograms_mean": {}}
        for key, values in self._samples.items():
            out["gauges"][key] = {"mean": sum(values) / len(values), "peak": max(values), "samples": len(values)}
        for key, names in _COUNTERS.items():
            out["totals"][key] = delta(names)
        for key, names in _HISTOGRAMS.items():
            total, count = delta(names, "_sum"), delta(names, "_count")
            out["histograms_mean"][key] = total / count if total is not None and count else None
        hits, queries = out["totals"]["prefix_cache_hits"], out["totals"]["prefix_cache_queries"]
        out["prefix_cache_hit_rate"] = hits / queries if hits is not None and queries else None
        cache = self._labels.get("vllm:cache_config_info")
        if cache:
            out["cache_config"] = {
                k: cache[k]
                for k in ("num_gpu_blocks", "block_size", "gpu_memory_utilization", "cache_dtype")
                if k in cache
            }
        return out
