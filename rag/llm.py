import structlog
from langchain_core.language_models import BaseChatModel

from config import settings

log = structlog.get_logger()


def get_llm() -> BaseChatModel:
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        log.info("llm_init", provider="ollama", model=settings.llm_model)
        return ChatOllama(
            model=settings.llm_model,
            base_url=settings.ollama_base_url,
            temperature=settings.temperature,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when LLM_PROVIDER=openai")
        log.info("llm_init", provider="openai", model=settings.llm_model)
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: '{settings.llm_provider}'. Choose 'ollama' or 'openai'.")
