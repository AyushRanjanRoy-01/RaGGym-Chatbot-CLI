"""PDF parsers.

``pymupdf` + ``pymupdf4llm`` convert a PDF into per-page Markdown, preserving
headings, code blocks, and tables — fast and dependency-light (no torch). For
richer figure/structure extraction, a Docling-based parser can be added behind
the same :class:`ParsedPage` contract (install the ``parse-advanced`` extra).
"""

from raggym.ingestion.parsers.pdf import ParsedPage, parse_pdf

__all__ = ["ParsedPage", "parse_pdf"]
