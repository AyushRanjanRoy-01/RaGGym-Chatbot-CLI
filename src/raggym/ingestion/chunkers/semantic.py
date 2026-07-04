"""Semantic (embedding-boundary) chunking via langchain-experimental.

Optional: install the ``semantic`` extra. Returns ``None`` when the dependency
or an embedding model is unavailable so callers can fall back to another strategy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from raggym.ingestion.parsers import ParsedPage


def semantic_chunks(
    pages: list[ParsedPage],
    *,
    book: str,
    source: str,
    embeddings: Any | None = None,
) -> list[Document] | None:
    """Split at semantic (embedding-similarity) boundaries; None if unavailable."""
    if embeddings is None:
        return None
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError:
        return None

    from langchain_core.documents import Document

    splitter = SemanticChunker(embeddings)
    docs: list[Document] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if piece:
                docs.append(
                    Document(
                        page_content=piece,
                        metadata={"book": book, "source": source, "page": page.page},
                    )
                )
    return docs
