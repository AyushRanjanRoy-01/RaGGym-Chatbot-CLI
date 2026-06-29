"""Chat-model factory.

Returns a LangChain chat model for the configured provider so the rest of the
codebase stays provider-agnostic:

* ``ollama``    — local server (default), e.g. ``llama3.2:3b`` (no API key)
* ``openai``    — e.g. ``gpt-5.4-mini`` (fast, good for hosted demos)
* ``anthropic`` — e.g. ``claude-sonnet-4-6`` (strong tutor/reviewer)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)

# Fallback model when the configured LLM_MODEL is still the ollama default but a
# cloud provider is selected.
_OLLAMA_DEFAULT = "llama3.2:3b"
_FALLBACK = {"openai": "gpt-5.4-mini", "anthropic": "claude-sonnet-4-6"}


def get_llm(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    provider = settings.llm_provider
    model = settings.llm_model
    if provider in _FALLBACK and model == _OLLAMA_DEFAULT:
        log.warning("llm_model_fallback", requested=model, using=_FALLBACK[provider])
        model = _FALLBACK[provider]

    log.info("llm_init", provider=provider, model=model)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model, api_key=settings.openai_api_key, temperature=settings.llm_temperature
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model, api_key=settings.anthropic_api_key, temperature=settings.llm_temperature
        )

    raise ValueError(f"Unknown llm_provider: {provider!r}")
