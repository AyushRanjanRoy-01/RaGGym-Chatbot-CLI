"""Retrieval engine [Phase 2].

RAG retrieval, each technique toggleable via settings:
    hybrid      dense vector search + local lexical fallback, fused with RRF
    multi_query LLM query expansion for recall
    reranker    cross-encoder re-scoring of candidates (flashrank)
"""

from raggym.retrieval.engine import RagRetriever

__all__ = ["RagRetriever"]
