"""Retrieval engine [Phase 2].

Advanced RAG retrieval, each technique toggleable via settings:
    hybrid      dense (vector) + sparse (BM25) fusion (Qdrant)
    multi_query LLM query expansion for recall
    reranker    cross-encoder re-scoring of candidates (flashrank)
"""

from raggym.retrieval.engine import RagRetriever, RetrievalSignals

__all__ = ["RagRetriever", "RetrievalSignals"]
