#!/usr/bin/env python3
"""
Ingest .txt documents into the vector store.

Usage:
    python ingest.py --docs ./docs
    python ingest.py --docs ./docs --reset
"""

import argparse
import shutil
import sys
from pathlib import Path

import structlog

from core.logging import setup_logging
from config import settings
from rag.chunkers import RecursiveChunker
from rag.embeddings import get_embeddings
from rag.loaders import TextLoader
from rag.vectorstore import get_vectorstore

setup_logging()
log = structlog.get_logger()


def _reset_vectorstore() -> None:
    path = Path(settings.qdrant_path if settings.vector_store == "qdrant" else settings.chroma_path)
    if path.exists():
        shutil.rmtree(path)
        log.info("vectorstore_cleared", path=str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store")
    parser.add_argument("--docs", required=True, help="Path to a .txt file or directory of .txt files")
    parser.add_argument("--reset", action="store_true", help="Wipe the vector store before ingesting")
    args = parser.parse_args()

    if args.reset:
        _reset_vectorstore()

    log.info("ingest_start", docs=args.docs, vector_store=settings.vector_store)

    try:
        documents = TextLoader().load(args.docs)
        chunks = RecursiveChunker().chunk(documents)

        embeddings = get_embeddings()
        store = get_vectorstore(embeddings)
        store.add_documents(chunks)

        log.info(
            "ingest_complete",
            source_documents=len(documents),
            chunks_stored=len(chunks),
            vector_store=settings.vector_store,
            collection=settings.collection_name,
        )

    except FileNotFoundError as exc:
        log.error("ingest_failed", error=str(exc))
        sys.exit(1)
    except Exception as exc:
        log.error("ingest_failed", error=str(exc), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
