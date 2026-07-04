"""Single source of truth for configuration.

All settings are read from environment variables / a ``.env`` file and validated
at startup. Override any field at runtime without code changes, e.g.::

    VECTOR_STORE=chroma raggym chat

The ``Settings`` object is a process-wide singleton via :func:`get_settings`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "openai", "anthropic"]
EmbedProvider = Literal["ollama", "openai", "fastembed"]
VectorStore = Literal["qdrant", "chroma"]
VisionProvider = Literal["ollama", "openai", "anthropic"]
ChunkStrategy = Literal["recursive", "fixed", "semantic"]


class Settings(BaseSettings):
    """Typed application settings. See ``.env.example`` for documentation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = "ollama"
    llm_model: str = "llama3.2:3b"
    llm_temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # ── Embeddings ───────────────────────────────────────────────────────────
    embed_provider: EmbedProvider = "ollama"
    embed_model: str = "nomic-embed-text"

    # ── Vector store ─────────────────────────────────────────────────────────
    vector_store: VectorStore = "qdrant"
    qdrant_collection: str = "raggym"
    qdrant_url: str | None = None  # None → local on-disk mode
    qdrant_api_key: str | None = None

    # ── Chunking ─────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    chunk_strategy: ChunkStrategy = "recursive"
    use_dedup: bool = True
    dedup_threshold: float = Field(default=0.9, ge=0, le=1)

    # ── Retrieval ────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, gt=0)
    use_hybrid: bool = True
    use_reranker: bool = False
    reranker_model: str = "ms-marco-MiniLM-L-12-v2"  # flashrank model id
    use_multi_query: bool = False

    # ── Chat agent ───────────────────────────────────────────────────────────
    # Corrective RAG: grade retrieved docs, rewrite the query and retry when the
    # context is weak (capped at max_retries). Off by default for speed/stability
    # with small local models.
    use_corrective: bool = False
    max_retries: int = Field(default=2, ge=0)

    # ── Vision / figure captioning ───────────────────────────────────────────
    enable_captioning: bool = False
    vision_provider: VisionProvider = "ollama"
    vision_model: str = "llava"

    # ── Paths ────────────────────────────────────────────────────────────────
    data_dir: Path = Path("./data")
    books_dir: Path = Path("./data/books")
    vectorstore_dir: Path = Path("./vectorstore")
    workspace_dir: Path = Path("./workspace")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_json: bool = False

    # ── App ──────────────────────────────────────────────────────────────────
    app_env: Literal["dev", "prod"] = "dev"

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        v = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return v

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_smaller_than_size(cls, v: int, info) -> int:
        size = info.data.get("chunk_size")
        if size is not None and v >= size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return v


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
