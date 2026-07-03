"""Retrieval engine: hybrid search + optional multi-query + optional rerank.

``RagRetriever`` wraps the configured vector store and applies the retrieval
techniques enabled in settings:

* **hybrid** — dense (vector) results are fused with a **BM25** sparse pass
  (proper IDF + TF-saturation + length-normalisation via ``rank-bm25``), working
  on both Qdrant and Chroma backends.
* **multi-query** — expands the question into paraphrases (LLM) and unions results.
* **fusion** — weighted **Reciprocal Rank Fusion** across all ranked lists.
* **rerank** — optional cross-encoder re-scoring of the fused pool.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from raggym.config import Settings, get_settings
from raggym.core import get_logger
from raggym.retrieval.rerank import rerank

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel
    from langchain_core.vectorstores import VectorStore

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens (len ≥ 2). BM25 IDF handles common terms."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def _doc_key(doc: Document) -> tuple:
    m = doc.metadata or {}
    return (m.get("source"), m.get("page"), doc.page_content[:64])


def _rrf_fuse(ranked_lists: list[tuple[float, list[Document]]], *, k: int = 60) -> list[Document]:
    """Weighted Reciprocal Rank Fusion.

    ``ranked_lists`` is a list of ``(weight, docs)`` where ``docs`` is ordered
    best-first. A document's fused score is ``Σ weight / (k + rank)``. Returns
    de-duplicated documents ordered by descending fused score.
    """
    scores: dict[tuple, float] = {}
    keep: dict[tuple, Document] = {}
    for weight, docs in ranked_lists:
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (k + rank + 1)
            keep.setdefault(key, doc)
    return sorted(keep.values(), key=lambda d: scores[_doc_key(d)], reverse=True)


class RagRetriever:
    """Stateful retriever over the book corpus. Call :meth:`close` when done."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        llm: BaseChatModel | None = None,
        vectorstore: VectorStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._llm = llm
        if vectorstore is not None:
            self.vs = vectorstore
        else:
            from raggym.embeddings import get_embeddings
            from raggym.vectorstore import get_vectorstore

            self.vs = get_vectorstore(get_embeddings(self.settings), self.settings)

    # ── query expansion ──────────────────────────────────────────────────────
    def _expand(self, query: str) -> list[str]:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        from raggym.llm import get_llm

        llm = self._llm or get_llm(self.settings)
        prompt = ChatPromptTemplate.from_template(
            "Generate 3 alternative search queries (different wording, same intent) "
            "for retrieving passages to answer the question. "
            "One per line, no numbering or commentary.\n\nQuestion: {q}"
        )
        text = (prompt | llm | StrOutputParser()).invoke({"q": query})
        variants = [line.strip("-•* \t") for line in text.splitlines() if line.strip()]
        return [query, *variants][:4]

    # ── sparse (BM25) ──────────────────────────────────────────────────────────
    def _all_documents(self) -> list[Document]:
        """Fetch the full corpus from the backend (Qdrant scroll or Chroma get)."""
        from langchain_core.documents import Document

        client = getattr(self.vs, "client", None)
        if client is not None and hasattr(client, "scroll"):
            collection = getattr(self.vs, "collection_name", self.settings.qdrant_collection)
            docs: list[Document] = []
            offset = None
            while True:
                records, offset = client.scroll(
                    collection_name=collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for record in records:
                    payload = record.payload or {}
                    docs.append(
                        Document(
                            page_content=payload.get("page_content") or "",
                            metadata=payload.get("metadata") or {},
                        )
                    )
                if offset is None:
                    break
            return docs

        # Chroma (or any store exposing .get())
        get = getattr(self.vs, "get", None)
        if callable(get):
            data = get(include=["documents", "metadatas"]) or {}
            texts = data.get("documents") or []
            metas = data.get("metadatas") or [{}] * len(texts)
            return [
                Document(page_content=t, metadata=m or {})
                for t, m in zip(texts, metas, strict=False)
            ]

        return []

    def _keyword_search(self, query: str, *, limit: int) -> list[Document]:
        """BM25 sparse retrieval over the corpus (backend-agnostic)."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            log.warning("bm25_unavailable", hint="pip install rank-bm25")
            return []

        corpus = [d for d in self._all_documents() if d.page_content.strip()]
        tokenized = [_tokenize(d.page_content) for d in corpus]
        pairs = [(d, t) for d, t in zip(corpus, tokenized, strict=False) if t]
        if not pairs:
            return []

        bm25 = BM25Okapi(
            [t for _, t in pairs], k1=self.settings.bm25_k1, b=self.settings.bm25_b
        )
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(range(len(pairs)), key=lambda i: scores[i], reverse=True)
        return [pairs[i][0] for i in ranked[:limit] if scores[i] > 0]

    # ── retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[Document]:
        s = self.settings
        queries = [query]
        if s.use_multi_query:
            try:
                queries = self._expand(query)
            except Exception as exc:  # noqa: BLE001 — never fail retrieval on expansion
                log.warning("multi_query_failed", error=str(exc))

        fetch_k = s.retrieval_top_k * s.overfetch_multiplier
        ranked_lists: list[tuple[float, list[Document]]] = []
        for q in queries:
            ranked_lists.append((s.hybrid_dense_weight, self.vs.similarity_search(q, k=fetch_k)))
            if s.use_hybrid:
                ranked_lists.append(
                    (s.hybrid_sparse_weight, self._keyword_search(q, limit=fetch_k))
                )

        fused = _rrf_fuse(ranked_lists, k=s.rrf_k)

        if s.use_reranker:
            docs = rerank(query, fused, model=s.reranker_model, top_n=s.retrieval_top_k)
        else:
            docs = fused[: s.retrieval_top_k]

        log.info(
            "retrieve_done",
            variants=len(queries),
            candidates=len(fused),
            returned=len(docs),
            hybrid=s.use_hybrid,
            reranked=s.use_reranker,
        )
        return docs

    def close(self) -> None:
        from raggym.vectorstore import close_vectorstore

        close_vectorstore(self.vs)
