from abc import ABC, abstractmethod
from functools import lru_cache

import structlog
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from config import settings

log = structlog.get_logger()


@lru_cache(maxsize=None)
def _get_local_qdrant_client(path: str):
    from qdrant_client import QdrantClient
    from pathlib import Path

    lock = Path(path) / ".lock"
    if lock.exists():
        lock.unlink()
        log.warning("qdrant_stale_lock_removed", path=path)

    client = QdrantClient(path=path)
    log.info("qdrant_init", mode="local", path=path)
    return client


@lru_cache(maxsize=None)
def _get_remote_qdrant_client(url: str, api_key: str):
    from qdrant_client import QdrantClient

    client = QdrantClient(url=url, api_key=api_key or None)
    log.info("qdrant_init", mode="remote", url=url)
    return client


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> None: ...

    @abstractmethod
    def as_retriever(self) -> VectorStoreRetriever: ...


class QdrantStore(BaseVectorStore):
    def __init__(self, embeddings: Embeddings):
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client.models import Distance, VectorParams

        if settings.qdrant_url:
            client = _get_remote_qdrant_client(
                settings.qdrant_url,
                settings.qdrant_api_key,
            )
        else:
            client = _get_local_qdrant_client(settings.qdrant_path)

        existing = {c.name for c in client.get_collections().collections}
        if settings.collection_name not in existing:
            client.create_collection(
                settings.collection_name,
                vectors_config=VectorParams(
                    size=settings.embed_dims,
                    distance=Distance.COSINE,
                ),
            )
            log.info("collection_created", name=settings.collection_name)

        self._store = QdrantVectorStore(
            client=client,
            collection_name=settings.collection_name,
            embedding=embeddings,
        )

    def add_documents(self, documents: list[Document]) -> None:
        self._store.add_documents(documents)
        log.info("documents_added", count=len(documents), store="qdrant")

    def as_retriever(self) -> VectorStoreRetriever:
        return self._store.as_retriever(search_kwargs={"k": settings.top_k})


class ChromaStore(BaseVectorStore):
    def __init__(self, embeddings: Embeddings):
        from langchain_chroma import Chroma
        self._store = Chroma(
            collection_name=settings.collection_name,
            embedding_function=embeddings,
            persist_directory=settings.chroma_path,
        )
        log.info("chroma_init", path=settings.chroma_path)

    def add_documents(self, documents: list[Document]) -> None:
        self._store.add_documents(documents)
        log.info("documents_added", count=len(documents), store="chroma")

    def as_retriever(self) -> VectorStoreRetriever:
        return self._store.as_retriever(search_kwargs={"k": settings.top_k})


def get_vectorstore(embeddings: Embeddings) -> BaseVectorStore:
    if settings.vector_store == "qdrant":
        return QdrantStore(embeddings)
    if settings.vector_store == "chroma":
        return ChromaStore(embeddings)
    raise ValueError(f"Unknown VECTOR_STORE: '{settings.vector_store}'. Choose 'qdrant' or 'chroma'.")
