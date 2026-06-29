"""Retrieval engine: hybrid search + optional multi-query + optional rerank.

``RagRetriever`` wraps the configured vector store and applies the retrieval
techniques enabled in settings:

* **hybrid** (dense + sparse) is handled by the Qdrant store itself.
* **multi-query** expands the question into paraphrases (LLM) and unions results.
* **rerank** re-scores the candidate pool with a cross-encoder and keeps top-k.
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

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "can",
    "define",
    "describe",
    "do",
    "does",
    "explain",
    "for",
    "how",
    "i",
    "in",
    "is",
    "me",
    "of",
    "on",
    "please",
    "tell",
    "the",
    "to",
    "what",
    "why",
    "you",
}


def _query_terms(query: str) -> list[str]:
    return [
        term
        for term in (match.group(0).lower() for match in _TOKEN_RE.finditer(query))
        if len(term) > 1 and term not in _STOPWORDS
    ]


def _lexical_score(terms: list[str], text: str) -> int:
    haystack = text.lower()
    return sum(haystack.count(term) for term in terms)


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

    def _keyword_search(self, query: str, *, limit: int) -> list[Document]:
        """Small local lexical fallback for exact terms when Qdrant runs in-process."""

        if self.settings.vector_store != "qdrant":
            return []

        terms = _query_terms(query)
        if not terms:
            return []

        client = getattr(self.vs, "client", None)
        collection = getattr(self.vs, "collection_name", self.settings.qdrant_collection)
        if client is None:
            return []

        from langchain_core.documents import Document

        offset = None
        matches: list[tuple[int, Document]] = []
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
                text = payload.get("page_content") or ""
                score = _lexical_score(terms, text)
                if score <= 0:
                    continue
                matches.append(
                    (
                        score,
                        Document(
                            page_content=text,
                            metadata=payload.get("metadata") or {},
                        ),
                    )
                )
            if offset is None:
                break

        matches.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in matches[:limit]]

    # ── retrieval ─────────────────────────────────────────────────────────────
    def retrieve(self, query: str) -> list[Document]:
        queries = [query]
        if self.settings.use_multi_query:
            try:
                queries = self._expand(query)
            except Exception as exc:  # noqa: BLE001 — never fail retrieval on expansion
                log.warning("multi_query_failed", error=str(exc))

        fetch_k = self.settings.retrieval_top_k * (4 if self.settings.use_reranker else 1)
        collected: dict[tuple, Document] = {}
        for q in queries:
            if self.settings.use_hybrid:
                for doc in self._keyword_search(q, limit=fetch_k):
                    key = (
                        doc.metadata.get("source"),
                        doc.metadata.get("page"),
                        doc.page_content[:64],
                    )
                    collected.setdefault(key, doc)
            for doc in self.vs.similarity_search(q, k=fetch_k):
                key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:64])
                collected.setdefault(key, doc)

        docs = list(collected.values())
        if self.settings.use_reranker:
            docs = rerank(
                query, docs, model=self.settings.reranker_model, top_n=self.settings.retrieval_top_k
            )
        else:
            docs = docs[: self.settings.retrieval_top_k]

        log.info(
            "retrieve_done",
            variants=len(queries),
            candidates=len(collected),
            returned=len(docs),
        )
        return docs

    def close(self) -> None:
        from raggym.vectorstore import close_vectorstore

        close_vectorstore(self.vs)
