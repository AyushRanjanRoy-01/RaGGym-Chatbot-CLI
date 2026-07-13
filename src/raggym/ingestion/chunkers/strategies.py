"""Chunking-strategy dispatcher.

Selects a strategy (``recursive`` heading-aware · ``fixed`` size · ``semantic``)
and tags every chunk with a ``strategy`` metadata field so strategies can be
compared. Unavailable ``semantic`` falls back to ``recursive``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from raggym.ingestion.parsers import ParsedPage

log = get_logger(__name__)


def chunk_pages_by_strategy(
    pages: list[ParsedPage],
    *,
    strategy: str = "recursive",
    book: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embeddings: Any | None = None,
) -> list[Document]:
    used = strategy
    docs: list[Document] | None = None

    if strategy == "fixed":
        from raggym.ingestion.chunkers.fixed import fixed_size_chunks

        docs = fixed_size_chunks(
            pages, book=book, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    elif strategy == "semantic":
        from raggym.ingestion.chunkers.semantic import semantic_chunks

        docs = semantic_chunks(pages, book=book, source=source, embeddings=embeddings)
        if docs is None:
            log.warning("semantic_chunking_unavailable", fallback="recursive")
            used = "recursive"

    if docs is None:  # recursive (default) or semantic fallback
        from raggym.ingestion.chunkers.markdown import chunk_pages

        docs = chunk_pages(
            pages, book=book, source=source, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

    for doc in docs:
        doc.metadata["strategy"] = used
    log.info("chunked", strategy=used, chunks=len(docs))
    return docs
