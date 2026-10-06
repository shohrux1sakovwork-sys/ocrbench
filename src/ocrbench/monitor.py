"""Background sampling of process memory and GPU usage during a run."""

from __future__ import annotations

import subprocess
import threading
import time

import psutil

_MB = 1024 * 1024


def _tree_rss_bytes(proc: psutil.Process) -> int:
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def _gpu_sample() -> dict[int, tuple[float, float]] | None:
    """GPU index -> (memory used MiB, utilization %), or None without nvidia-smi."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    sample = {}
    for line in out.strip().splitlines():
        idx, mem, util = (field.strip() for field in line.split(","))
        sample[int(idx)] = (float(mem), float(util))
    return sample


class ResourceMonitor:
    """Samples RSS of this process and its children (render workers).

    GPU numbers come from nvidia-smi and cover every process on the GPU,
    including other users' jobs, so they are reported next to a baseline
    taken before the run.
    """

    def __init__(
        self,
        interval_s: float = 0.25,
        gpu_interval_s: float = 1.0,
        gpu: bool = True,
        gpus: tuple[int, ...] | None = None,
    ):
        self.interval_s = interval_s
        self.gpu_interval_s = gpu_interval_s
        self.gpu = gpu
        self.gpus = gpus
        self._proc = psutil.Process()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._rss_samples: list[int] = []
        self._gpu_baseline: dict[int, tuple[float, float]] | None = None
        self._gpu_samples: list[dict[int, tuple[float, float]]] = []

    def rss_mb(self) -> float:
        return _tree_rss_bytes(self._proc) / _MB

    def start(self) -> None:
        self._rss_baseline = _tree_rss_bytes(self._proc)
        if self.gpu:
            self._gpu_baseline = _gpu_sample()
        self._thread.start()

    def _loop(self) -> None:
        next_gpu = 0.0
        while not self._stop.is_set():
            self._rss_samples.append(_tree_rss_bytes(self._proc))
            now = time.monotonic()
            if self._gpu_baseline is not None and now >= next_gpu:
                sample = _gpu_sample()
                if sample:
                    self._gpu_samples.append(sample)
                next_gpu = now + self.gpu_interval_s
            self._stop.wait(self.interval_s)

    def stop(self) -> dict:
        self._stop.set()
        self._thread.join()
        rss = self._rss_samples or [self._rss_baseline]
        summary: dict = {
            "rss_baseline_mb": self._rss_baseline / _MB,
            "rss_peak_mb": max(rss) / _MB,
            "rss_mean_mb": sum(rss) / len(rss) / _MB,
            "samples": len(rss),
        }
        if self._gpu_baseline and self._gpu_samples:
            gpus = {}
            for idx, (base_mem, base_util) in self._gpu_baseline.items():
                if self.gpus is not None and idx not in self.gpus:
                    continue
                mems = [s[idx][0] for s in self._gpu_samples if idx in s]
                utils = [s[idx][1] for s in self._gpu_samples if idx in s]
                gpus[str(idx)] = {
                    "memory_baseline_mib": base_mem,
                    "memory_peak_mib": max(mems),
                    "utilization_baseline_pct": base_util,
                    "utilization_mean_pct": sum(utils) / len(utils),
                    "utilization_peak_pct": max(utils),
                }
            summary["gpus"] = gpus
        return summary
