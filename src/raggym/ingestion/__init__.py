"""Ingestion pipeline [Phase 1].

PDF (and future formats) → typed elements (prose, code, table, figure,
glossary, callout/box-text) → element-aware chunking → embeddings → vector
store, with per-book metadata (book, chapter, page, element_type, language).

Submodules (planned):
    parsers/      Docling (code/tables/figures/reading-order) + PyMuPDF fallback
    chunkers/     code kept whole, tables structured, prose recursive
    captioner.py  pluggable vision stage (off by default)
    pipeline.py   orchestrates parse → chunk → embed → upsert across many books
"""
