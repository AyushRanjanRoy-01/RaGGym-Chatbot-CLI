"""Ingestion pipeline [Phase 1].

PDFs are parsed to per-page Markdown, split into heading-aware chunks, embedded,
and written to the configured vector store with per-book metadata (book,
section, page, source).

    parsers/      pymupdf4llm → per-page Markdown (Docling optional upgrade)
    chunkers/     markdown-aware recursive splitting, page + heading metadata
    captioning.py optional vision captions for visual-heavy pages
    pipeline.py   orchestrates parse → chunk → embed → upsert across many books
"""

from raggym.ingestion.pipeline import ingest_path

__all__ = ["ingest_path"]
