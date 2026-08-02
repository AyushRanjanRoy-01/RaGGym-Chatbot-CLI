"""Compare chunking strategies over a document (counts + size stats)."""

from __future__ import annotations

from statistics import mean
from typing import TYPE_CHECKING, Any

from raggym.core import get_logger

if TYPE_CHECKING:
    from raggym.ingestion.parsers import ParsedPage

log = get_logger(__name__)


def compare_chunking_strategies(
    pages: list[ParsedPage],
    *,
    book: str = "doc",
    source: str = "doc",
    strategies: tuple[str, ...] = ("recursive", "fixed"),
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embeddings: Any | None = None,
) -> list[dict]:
    """Chunk with each strategy and return per-strategy stats (count + sizes)."""
    from raggym.ingestion.chunkers import chunk_pages_by_strategy

    rows: list[dict] = []
    for strategy in strategies:
        docs = chunk_pages_by_strategy(
            pages,
            strategy=strategy,
            book=book,
            source=source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embeddings=embeddings,
        )
        sizes = [len(d.page_content) for d in docs]
        rows.append(
            {
                "strategy": strategy,
                "chunks": len(docs),
                "avg_chars": round(mean(sizes), 1) if sizes else 0.0,
                "min_chars": min(sizes, default=0),
                "max_chars": max(sizes, default=0),
            }
        )
    log.info("chunk_report", strategies=list(strategies))
    return rows
