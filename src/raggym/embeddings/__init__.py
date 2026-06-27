"""Embeddings factory [Phase 1].

Returns the configured embedding model (ollama nomic-embed-text by default,
openai or fastembed as alternatives). The same model MUST be used at ingest and
query time.
"""

from raggym.embeddings.factory import get_embeddings

__all__ = ["get_embeddings"]
