"""Embedding-model factory.

Returns a LangChain ``Embeddings`` for the configured provider:

* ``ollama``    — local server (default), e.g. ``nomic-embed-text`` (8192-token ctx)
* ``openai``    — ``text-embedding-3-small`` (upgrade path; needs an API key)
* ``fastembed`` — local ONNX models, **no server and no API key** (zero-setup)

The embedding model MUST be identical at ingest time and query time — changing
it requires a full re-ingest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

log = get_logger(__name__)

# Used when the configured EMBED_MODEL is the ollama default but the provider is
# fastembed (which needs a HuggingFace-style model id).
_FASTEMBED_DEFAULT = "BAAI/bge-small-en-v1.5"


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    settings = settings or get_settings()
    provider = settings.embed_provider

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        log.info("embeddings_init", provider=provider, model=settings.embed_model)
        return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        log.info("embeddings_init", provider=provider, model=settings.embed_model)
        return OpenAIEmbeddings(model=settings.embed_model, api_key=settings.openai_api_key)

    if provider == "fastembed":
        from raggym.embeddings._fastembed import FastEmbedEmbeddings

        model = settings.embed_model
        if model == "nomic-embed-text":  # the ollama default — not a fastembed id
            log.warning("embed_model_fallback", requested=model, using=_FASTEMBED_DEFAULT)
            model = _FASTEMBED_DEFAULT
        log.info("embeddings_init", provider=provider, model=model)
        return FastEmbedEmbeddings(model_name=model)

    raise ValueError(f"Unknown embed_provider: {provider!r}")
