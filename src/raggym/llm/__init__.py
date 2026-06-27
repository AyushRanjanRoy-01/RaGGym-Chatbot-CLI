"""LLM provider factory [Phase 2].

Returns a LangChain chat model for the configured provider — ollama (default),
openai, or anthropic — so the rest of the codebase is provider-agnostic.
"""

from raggym.llm.factory import get_llm

__all__ = ["get_llm"]
