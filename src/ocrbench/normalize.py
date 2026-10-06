"""Text normalization applied to references and predictions before scoring.

The ground truth is Markdown with bracketed annotations ([EMBLEM],
[HANDWRITTEN: ...]); OCR models emit Markdown/HTML with their own annotation
tags (<img>, <signature>, <page_number>). Normalization removes markup and
non-textual annotations from both sides with the same rules, so CER/WER
measure recognized text rather than formatting conventions.
"""

from __future__ import annotations

import html
import re
import unicodedata

_HANDWRITTEN = re.compile(r"\[HANDWRITTEN:\s*([^\]]*)\]")
_GT_ANNOTATION = re.compile(
    r"\[(?:EMBLEM|SIGNATURE|QR|LOGO|STAMP|WATERMARK|BARCODE)(?::[^\]]*)?\]|\[\s*\]"
)
# Descriptions of non-text elements; their content is not document text.
_ANNOTATION_BLOCK = re.compile(
    r"<(img|signature|watermark|stamp|logo|qr|barcode)\b[^>]*>.*?</\1\s*>",
    re.S | re.I,
)
# Markdown image placeholders, e.g. ![image](image_1.png).
_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
# LaTeX text wrappers and inline math delimiters around plain text.
_LATEX_TEXT = re.compile(r"\\(?:text|textbf|textit|mathrm|mathbf)\{([^{}]*)\}")
_MATH_DELIM = re.compile(r"\$\$?([^$\n]{1,300}?)\$\$?|\\\((.{1,300}?)\\\)", re.S)
_BREAK_TAG = re.compile(r"<br\s*/?>|</(?:p|div|tr|li|h[1-6])\s*>", re.I)
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_TABLE_RULE = re.compile(r"^[\s|:-]*-{3,}[\s|:-]*$", re.M)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)
_WHITESPACE = re.compile(r"\s+")

_CHAR_MAP = str.maketrans(
    {
        **dict.fromkeys("‘’ʻʼ`´′ʹ", "'"),
        **dict.fromkeys("«»“”„‟″", '"'),
        **dict.fromkeys("‐‑‒–—―−", "-"),
        " ": " ",
        "…": "...",
        "*": None,
        "|": " ",
        "\\": None,
    }
)


def normalize(text: str, *, lowercase: bool = False) -> str:
    t = unicodedata.normalize("NFC", text)
    t = _HANDWRITTEN.sub(r" \1 ", t)
    t = _GT_ANNOTATION.sub(" ", t)
    t = _ANNOTATION_BLOCK.sub(" ", t)
    t = _MD_IMAGE.sub(" ", t)
    t = _LATEX_TEXT.sub(r"\1", t)
    t = _MATH_DELIM.sub(lambda m: m.group(1) or m.group(2), t)
    t = _BREAK_TAG.sub("\n", t)
    t = _TAG.sub(" ", t)
    t = html.unescape(t)
    t = _TABLE_RULE.sub(" ", t)
    t = _HEADING.sub("", t)
    t = t.translate(_CHAR_MAP)
    if lowercase:
        t = t.lower()
    return _WHITESPACE.sub(" ", t).strip()


def normalize_raw(text: str) -> str:
    """Only Unicode composition and whitespace; markup counts as errors."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()
