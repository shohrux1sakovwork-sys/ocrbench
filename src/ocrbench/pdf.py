"""Page rendering, executed in worker processes.

PyMuPDF is not thread-safe, so pages are rendered in a process pool. Each
worker keeps its most recent document open; pages are rendered one at a time,
so memory does not grow with the size of the PDF.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import pymupdf

_open_doc: tuple[str, pymupdf.Document] | None = None


@dataclass(frozen=True)
class RenderOptions:
    dpi: int = 150
    max_side: int = 1800
    image_format: str = "png"
    jpeg_quality: int = 92


def _document(path: str) -> pymupdf.Document:
    global _open_doc
    if _open_doc is None or _open_doc[0] != path:
        if _open_doc is not None:
            _open_doc[1].close()
        _open_doc = (path, pymupdf.open(path))
    return _open_doc[1]


def render_page(path: str, page: int, opts: RenderOptions) -> dict:
    """Render 1-based ``page`` of ``path`` to a base64 data URL."""
    start = time.perf_counter()
    doc = _document(path)
    if not 1 <= page <= doc.page_count:
        raise IndexError(f"page {page} out of range; document has {doc.page_count} pages")
    pdf_page = doc.load_page(page - 1)
    zoom = opts.dpi / 72
    longest = max(pdf_page.rect.width, pdf_page.rect.height) * zoom
    if opts.max_side and longest > opts.max_side:
        zoom *= opts.max_side / longest
    pix = pdf_page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    if opts.image_format == "jpeg":
        data, mime = pix.tobytes("jpeg", jpg_quality=opts.jpeg_quality), "image/jpeg"
    else:
        data, mime = pix.tobytes("png"), "image/png"
    width, height = pix.width, pix.height
    del pix
    return {
        "image_url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
        "width": width,
        "height": height,
        "image_bytes": len(data),
        "render_s": time.perf_counter() - start,
    }


def page_count(path: str) -> int:
    with pymupdf.open(path) as doc:
        return doc.page_count
