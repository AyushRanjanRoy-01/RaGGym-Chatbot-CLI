"""PDF → per-page Markdown via pymupdf4llm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from raggym.core import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class ParsedPage:
    """One parsed page. ``page`` is 1-based for human-friendly citations."""

    page: int
    text: str  # Markdown


def parse_pdf(path: str | Path, *, max_pages: int | None = None) -> list[ParsedPage]:
    """Parse a PDF into a list of :class:`ParsedPage` (Markdown per page).

    Args:
        max_pages: if set, parse only the first N pages (fast path for tests).

    Empty/whitespace-only pages are dropped.
    """
    import pymupdf4llm

    path = Path(path)
    log.info("parse_pdf_start", file=path.name, max_pages=max_pages)

    # page_chunks=True → one dict per page with keys including 'text' and 'metadata'.
    kwargs: dict = {"page_chunks": True, "show_progress": False}
    if max_pages:
        kwargs["pages"] = list(range(max_pages))
    raw_pages = pymupdf4llm.to_markdown(str(path), **kwargs)

    pages: list[ParsedPage] = []
    for i, chunk in enumerate(raw_pages):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        meta = chunk.get("metadata") or {}
        # pymupdf 'page' is 0-based; fall back to enumeration order.
        page_no = meta.get("page")
        page_no = (page_no + 1) if isinstance(page_no, int) else (i + 1)
        pages.append(ParsedPage(page=page_no, text=text))

    log.info("parse_pdf_done", file=path.name, pages=len(pages))
    return pages
