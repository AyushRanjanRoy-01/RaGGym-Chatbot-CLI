from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai"] = "ollama" # set here
    llm_model: str = "llama3.2:3b"
    ollama_base_url: str = "http://localhost:11434"
    # openai_api_key: str = ""
    temperature: float = 0.1

    # ── Embeddings ────────────────────────────────────────────────────────────
    embed_provider: Literal["ollama", "openai"] = "ollama"
    embed_model: str = "nomic-embed-text"
    embed_dims: int = 768

    # ── Vector store ──────────────────────────────────────────────────────────
    vector_store: Literal["qdrant", "chroma"] = "qdrant"
    collection_name: str = "financial_docs"

    qdrant_path: str = "./vectorstore"
    qdrant_url: str = ""        # set for remote / Qdrant Cloud
    qdrant_api_key: str = ""

    chroma_path: str = "./vectorstore"

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ─────────────────────────────────────────────────────────────
    top_k: int = 5

    # ── App ───────────────────────────────────────────────────────────────────
    app_title: str = "Financial RAG Assistant"
    app_description: str = "Ask questions about your financial documents."
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
