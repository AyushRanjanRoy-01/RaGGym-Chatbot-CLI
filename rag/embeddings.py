import structlog
from langchain_core.embeddings import Embeddings

from config import settings

log = structlog.get_logger()


def get_embeddings() -> Embeddings:
    if settings.embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        log.info("embeddings_init", provider="ollama", model=settings.embed_model)
        return OllamaEmbeddings(
            model=settings.embed_model,
            base_url=settings.ollama_base_url,
        )

    if settings.embed_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when EMBED_PROVIDER=openai")
        log.info("embeddings_init", provider="openai", model=settings.embed_model)
        return OpenAIEmbeddings(
            model=settings.embed_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"Unknown EMBED_PROVIDER: '{settings.embed_provider}'. Choose 'ollama' or 'openai'.")
