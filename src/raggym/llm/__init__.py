"""LLM provider factory [Phase 2].

Returns a LangChain chat model for the configured provider — ollama (default),
openai, or anthropic — so the rest of the codebase is provider-agnostic.
"""
