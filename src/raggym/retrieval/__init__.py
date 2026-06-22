"""Retrieval engine [Phase 2].

Advanced RAG retrieval, each technique toggleable via settings:
    hybrid      dense (vector) + sparse (BM25) fusion
    reranker    cross-encoder re-scoring of candidates (e.g. bge-reranker)
    multi_query LLM query expansion for recall
    router      picks a strategy per query
"""
