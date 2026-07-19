"""Document parsers.

``pymupdf4llm`` converts PDFs to per-page Markdown; ``text`` loads Markdown /
plain-text / HTML. :func:`load_document` dispatches by file extension, all behind
the same :class:`ParsedPage` contract.
"""

from __future__ import annotations

from pathlib import Path

from raggym.ingestion.parsers.pdf import ParsedPage, parse_pdf
from raggym.ingestion.parsers.text import parse_html, parse_text_file

_LOADERS = {
    ".md": parse_text_file,
    ".markdown": parse_text_file,
    ".txt": parse_text_file,
    ".html": parse_html,
    ".htm": parse_html,
}
SUPPORTED_SUFFIXES = (".pdf", *_LOADERS)

__all__ = ["ParsedPage", "SUPPORTED_SUFFIXES", "load_document", "parse_pdf"]


def load_document(path: str | Path, *, max_pages: int | None = None) -> list[ParsedPage]:
    """Parse any supported document into pages (dispatch by extension)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, max_pages=max_pages)
    loader = _LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported file type: {suffix}")
    return loader(path)
