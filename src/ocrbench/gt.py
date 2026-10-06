"""Ground-truth loading and document matching.

Matching works on directory listings only, so a PDF is never opened unless it
has a ground-truth entry.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Entries with these statuses exist in the ground-truth file but their text is
# not a reference transcription: "pending" has no text yet, and the raw text
# layer is the PDF's embedded text rather than a checked transcription.
NON_REFERENCE_STATUSES = frozenset({"pending", "raw_text_layer_not_ground_truth"})


def gt_key(filename: str) -> str:
    """Return the sanitized name the ground-truth file uses for ``filename``.

    The ground truth stores names with every character other than letters,
    digits, "_" and "-" replaced by "_" after Unicode decomposition, so
    ``ЎРҚ-1166.pdf`` becomes ``У_РҚ-1166.pdf`` and ``08-1-2224++.pdf`` becomes
    ``08-1-2224__.pdf``. Applying the rule to either form gives the same key.
    """
    stem, ext = os.path.splitext(filename)
    stem = re.sub(r"[^\w-]", "_", unicodedata.normalize("NFD", stem))
    return unicodedata.normalize("NFC", stem) + ext.lower()


@dataclass(frozen=True)
class GTDocument:
    name: str
    status: str
    page_count: int
    # 1-based page number -> reference text, for pages with non-empty text.
    pages: dict[int, str]

    @property
    def has_reference(self) -> bool:
        return self.status not in NON_REFERENCE_STATUSES and bool(self.pages)

    def reference(self, page: int) -> str | None:
        return self.pages.get(page) if self.has_reference else None


@dataclass(frozen=True)
class MatchedDocument:
    gt: GTDocument
    path: Path


@dataclass(frozen=True)
class MatchResult:
    matched: list[MatchedDocument]
    without_gt: list[Path]
    missing_files: list[GTDocument]


def load_ground_truth(path: str | Path) -> dict[str, GTDocument]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    docs: dict[str, GTDocument] = {}
    for entry in entries:
        raw_pages = entry.get("pages") or []
        pages = {
            int(p["page"]): p["text"]
            for p in raw_pages
            if isinstance(p.get("text"), str) and p["text"].strip()
        }
        doc = GTDocument(
            name=entry["file"],
            status=entry.get("status", ""),
            page_count=int(entry.get("page_count") or len(raw_pages)),
            pages=pages,
        )
        key = gt_key(doc.name)
        if key in docs:
            raise ValueError(f"duplicate ground-truth entry for {doc.name!r}")
        docs[key] = doc
    return docs


def match_documents(gt: dict[str, GTDocument], folder: str | Path) -> MatchResult:
    matched: list[MatchedDocument] = []
    without_gt: list[Path] = []
    matched_by: dict[str, str] = {}
    for entry in sorted(os.scandir(folder), key=lambda e: e.name):
        if not entry.is_file():
            continue
        doc = gt.get(gt_key(entry.name))
        if doc is None:
            without_gt.append(Path(entry.path))
            continue
        if doc.name in matched_by:
            raise ValueError(
                f"{entry.name!r} and {matched_by[doc.name]!r} both match "
                f"ground-truth entry {doc.name!r}"
            )
        matched_by[doc.name] = entry.name
        matched.append(MatchedDocument(doc, Path(entry.path)))
    missing = [d for d in gt.values() if d.name not in matched_by]
    return MatchResult(matched, without_gt, missing)
