"""Accuracy metrics (CER/WER) and the statistics used for timing/memory."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from rapidfuzz.distance import Levenshtein

from .normalize import normalize, normalize_raw


@dataclass
class EditCounts:
    errors: int = 0
    ref_len: int = 0
    hyp_len: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    def __iadd__(self, other: EditCounts) -> EditCounts:
        for name, value in asdict(other).items():
            setattr(self, name, getattr(self, name) + value)
        return self

    @property
    def rate(self) -> float | None:
        return self.errors / self.ref_len if self.ref_len else None

    @classmethod
    def from_dict(cls, data: dict) -> EditCounts:
        return cls(**data)


def edit_counts(ref: Sequence, hyp: Sequence) -> EditCounts:
    """Levenshtein alignment of ``hyp`` against ``ref`` (chars or word lists)."""
    ops = Counter(tag for tag, _, _ in Levenshtein.editops(ref, hyp).as_list())
    return EditCounts(
        errors=sum(ops.values()),
        ref_len=len(ref),
        hyp_len=len(hyp),
        substitutions=ops["replace"],
        insertions=ops["insert"],
        deletions=ops["delete"],
    )


def score_page(reference: str, prediction: str, *, lowercase: bool = False) -> dict:
    """CER/WER on normalized text, plus raw CER/WER where markup counts."""
    ref, hyp = normalize(reference, lowercase=lowercase), normalize(prediction, lowercase=lowercase)
    raw_ref, raw_hyp = normalize_raw(reference), normalize_raw(prediction)
    chars = edit_counts(ref, hyp)
    words = edit_counts(ref.split(), hyp.split())
    raw_chars = edit_counts(raw_ref, raw_hyp)
    raw_words = edit_counts(raw_ref.split(), raw_hyp.split())
    return {
        "cer": chars.rate,
        "wer": words.rate,
        "raw_cer": raw_chars.rate,
        "raw_wer": raw_words.rate,
        "chars": asdict(chars),
        "words": asdict(words),
        "raw_chars": asdict(raw_chars),
        "raw_words": asdict(raw_words),
    }


def percentiles(values: Sequence[float], ps: Sequence[int] = (50, 90, 95, 99)) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    out = {"min": ordered[0], "mean": sum(ordered) / len(ordered), "max": ordered[-1]}
    for p in ps:
        # Linear interpolation between closest ranks.
        pos = (len(ordered) - 1) * p / 100
        lo, hi = math.floor(pos), math.ceil(pos)
        out[f"p{p}"] = ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return out


def linear_fit(xs: Sequence[float], ys: Sequence[float]) -> dict | None:
    """Least-squares y = intercept + slope * x."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - intercept - slope * x) ** 2 for x, y in zip(xs, ys))
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n}


def complexity_fit(sizes: Sequence[float], costs: Sequence[float]) -> dict | None:
    """Fit cost ≈ c · size^k on a log-log scale; k is the empirical order.

    k ≈ 0 means cost does not depend on size, k ≈ 1 linear, k ≈ 2 quadratic.
    A low r² means size explains little of the variation in cost. Returns
    None when sizes span less than 1.5x, where the exponent is meaningless.
    """
    pairs = [(math.log(s), math.log(c)) for s, c in zip(sizes, costs) if s > 0 and c > 0]
    if not pairs or max(p[0] for p in pairs) - min(p[0] for p in pairs) < math.log(1.5):
        return None
    fit = linear_fit([p[0] for p in pairs], [p[1] for p in pairs])
    if fit is None:
        return None
    k = fit["slope"]
    return {
        "exponent": k,
        "r2": fit["r2"],
        "n": fit["n"],
        "order": _order_label(k),
    }


def _order_label(k: float) -> str:
    if k < 0.25:
        return "~O(1)"
    if k < 0.75:
        return "sub-linear"
    if k < 1.25:
        return "~O(n)"
    if k < 1.75:
        return "super-linear"
    return "~O(n^2) or worse"
