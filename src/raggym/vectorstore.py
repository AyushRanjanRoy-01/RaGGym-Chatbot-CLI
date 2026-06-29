"""Vector store factory for the RAGGym corpus."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

log = get_logger(__name__)


def _embedding_size(embeddings: Embeddings) -> int:
    return len(embeddings.embed_query("dimension probe"))


def _get_qdrant_store(
    embeddings: Embeddings,
    settings: Settings,
    *,
    create: bool,
    recreate: bool,
) -> VectorStore:
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    if settings.qdrant_url:
        client = QdrantClient(url=settings.qdrant_url)
        log.info("qdrant_init", mode="remote", url=settings.qdrant_url)
    else:
        settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        client = QdrantClient(path=str(settings.vectorstore_dir))
        log.info("qdrant_init", mode="local", path=str(settings.vectorstore_dir))

    collection = settings.qdrant_collection
    existing = {item.name for item in client.get_collections().collections}

    if recreate and collection in existing:
        client.delete_collection(collection)
        existing.remove(collection)
        log.info("qdrant_collection_deleted", collection=collection)

    if create and collection not in existing:
        client.create_collection(
            collection,
            vectors_config=VectorParams(
                size=_embedding_size(embeddings),
                distance=Distance.COSINE,
            ),
        )
        log.info("qdrant_collection_created", collection=collection)

    return QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=embeddings,
    )


def _get_chroma_store(
    embeddings: Embeddings,
    settings: Settings,
    *,
    recreate: bool,
) -> VectorStore:
    from langchain_chroma import Chroma

    chroma_path = settings.vectorstore_dir / "chroma"
    if recreate and chroma_path.exists():
        shutil.rmtree(chroma_path)
        log.info("chroma_store_deleted", path=str(chroma_path))

    chroma_path.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=settings.qdrant_collection,
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )


def get_vectorstore(
    embeddings: Embeddings,
    settings: Settings | None = None,
    *,
    create: bool = False,
    recreate: bool = False,
) -> VectorStore:
    """Return the configured vector store.

    ``create`` is used by ingestion, while query paths can open an existing
    collection without deleting anything.
    """

    settings = settings or get_settings()

    if settings.vector_store == "qdrant":
        return _get_qdrant_store(embeddings, settings, create=create, recreate=recreate)
    if settings.vector_store == "chroma":
        return _get_chroma_store(embeddings, settings, recreate=recreate)

    raise ValueError(f"Unknown vector_store: {settings.vector_store!r}")


def close_vectorstore(vectorstore: VectorStore | None) -> None:
    """Close underlying clients when the store exposes one."""

    if vectorstore is None:
        return

    client = getattr(vectorstore, "client", None)
    close = getattr(client, "close", None)
    if callable(close):
        close()
