"""Minimal LangChain ``Embeddings`` wrapper over the ``fastembed`` library.

Keeps the zero-setup local embedding path off the sunset ``langchain-community``
package. FastEmbed runs quantized ONNX models locally — no server, no API key.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings


class FastEmbedEmbeddings(Embeddings):
    """LangChain-compatible embeddings backed by ``fastembed.TextEmbedding``."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()
