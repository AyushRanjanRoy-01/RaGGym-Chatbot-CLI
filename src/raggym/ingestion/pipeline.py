"""Ingestion orchestration: PDF(s) → parse → chunk → embed → vector store.

Handles a single file or a whole directory of PDFs (the multi-book corpus),
attaching per-book metadata so retrieval can cite and filter by source.
"""

from __future__ import annotations

import time
from pathlib import Path

from raggym.config import Settings, get_settings
from raggym.core import get_logger
from raggym.embeddings import get_embeddings
from raggym.ingestion.cache import load_pages_cached
from raggym.ingestion.captioning import caption_pdf_visual_pages
from raggym.ingestion.chunkers import chunk_pages_by_strategy
from raggym.ingestion.dedup import dedupe_chunks
from raggym.ingestion.parsers import SUPPORTED_SUFFIXES, load_document
from raggym.vectorstore import close_vectorstore, get_vectorstore

log = get_logger(__name__)


def _discover_docs(target: Path) -> list[Path]:
    if target.is_file():
        return [target] if target.suffix.lower() in SUPPORTED_SUFFIXES else []
    return sorted(p for p in target.glob("*.*") if p.suffix.lower() in SUPPORTED_SUFFIXES)


def ingest_path(
    path: str | Path | None = None,
    *,
    settings: Settings | None = None,
    limit_pages: int | None = None,
    recreate: bool = False,
    batch_size: int = 128,
) -> dict:
    """Ingest a PDF or a directory of PDFs into the vector store.

    Args:
        path: PDF file or directory. Defaults to ``settings.books_dir``.
        limit_pages: ingest only the first N pages per book (quick tests).
        recreate: drop and rebuild the collection before ingesting.
        batch_size: number of chunks embedded/upserted per batch.

    Returns:
        Summary dict: ``{"books", "chunks", "files": [...]}``.
    """
    settings = settings or get_settings()
    target = Path(path) if path else settings.books_dir

    paths = _discover_docs(target)
    if not paths:
        log.warning("no_documents_found", path=str(target))
        return {"books": 0, "chunks": 0, "files": []}

    embeddings = get_embeddings(settings)
    vs = get_vectorstore(embeddings, settings, create=True, recreate=recreate)

    files: list[dict] = []
    total_chunks = 0
    try:
        for src in paths:
            t0 = time.perf_counter()
            if settings.use_parse_cache:
                pages = load_pages_cached(
                    src,
                    cache_dir=settings.parse_cache_dir,
                    loader=load_document,
                    max_pages=limit_pages,
                )
            else:
                pages = load_document(src, max_pages=limit_pages)
            visual_captions = (
                caption_pdf_visual_pages(src, settings=settings, max_pages=limit_pages)
                if src.suffix.lower() == ".pdf"
                else []
            )
            docs = chunk_pages_by_strategy(
                [*pages, *visual_captions],
                strategy=settings.chunk_strategy,
                book=src.stem,
                source=src.name,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                embeddings=embeddings,
            )
            if settings.use_dedup:
                docs = dedupe_chunks(docs, threshold=settings.dedup_threshold)
            for i in range(0, len(docs), batch_size):
                vs.add_documents(docs[i : i + batch_size])

            elapsed = round(time.perf_counter() - t0, 1)
            total_chunks += len(docs)
            files.append(
                {
                    "book": src.name,
                    "pages": len(pages),
                    "visual_captions": len(visual_captions),
                    "chunks": len(docs),
                    "seconds": elapsed,
                }
            )
            log.info(
                "book_ingested",
                book=src.name,
                pages=len(pages),
                visual_captions=len(visual_captions),
                chunks=len(docs),
                seconds=elapsed,
            )
    finally:
        close_vectorstore(vs)

    log.info("ingest_complete", books=len(paths), chunks=total_chunks)
    return {"books": len(paths), "chunks": total_chunks, "files": files}
