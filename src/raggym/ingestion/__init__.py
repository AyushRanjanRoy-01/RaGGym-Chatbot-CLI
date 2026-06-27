"""Ingestion pipeline [Phase 1].

PDF (and future formats) → typed elements (prose, code, table, figure,
glossary, callout/box-text) → element-aware chunking → embeddings → vector
store, with per-book metadata (book, chapter/section, page, source).

    parsers/      pymupdf4llm → per-page Markdown (Docling optional upgrade)
    chunkers/     markdown-aware recursive splitting, page + heading metadata
    captioner.py  pluggable vision stage (off by default)        [Phase 4]
    pipeline.py   orchestrates parse → chunk → embed → upsert across many books
"""

from raggym.ingestion.pipeline import ingest_path

__all__ = ["ingest_path"]
