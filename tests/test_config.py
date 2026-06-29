"""Smoke tests for the configuration and logging foundation."""

import pytest

from raggym import __version__
from raggym.config import Settings, get_settings


def test_version_is_set():
    assert __version__
    assert isinstance(__version__, str)


def test_settings_load_with_defaults(monkeypatch, tmp_path):
    # Ignore any developer .env AND any exported env vars so defaults are
    # deterministic regardless of the shell the tests run in.
    monkeypatch.chdir(tmp_path)
    for key in (
        "LLM_PROVIDER", "LLM_MODEL", "EMBED_PROVIDER", "EMBED_MODEL",
        "VECTOR_STORE", "CHUNK_SIZE", "CHUNK_OVERLAP", "USE_HYBRID",
    ):
        monkeypatch.delenv(key, raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.llm_provider == "ollama"
    assert settings.embed_model == "nomic-embed-text"
    assert settings.vector_store == "qdrant"
    assert settings.chunk_size == 1000
    assert settings.chunk_overlap == 200


def test_settings_is_singleton():
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        Settings(chunk_size=500, chunk_overlap=500, _env_file=None)


def test_invalid_log_level_rejected():
    with pytest.raises(ValueError):
        Settings(log_level="VERBOSE", _env_file=None)


def test_log_level_normalised_to_upper():
    s = Settings(log_level="debug", _env_file=None)
    assert s.log_level == "DEBUG"


def test_app_mode_default_and_alias(monkeypatch):
    monkeypatch.delenv("RAGGYM_MODE", raising=False)
    monkeypatch.delenv("APP_MODE", raising=False)
    assert Settings(_env_file=None).app_mode == "local"
    monkeypatch.setenv("RAGGYM_MODE", "cloud")
    assert Settings(_env_file=None).app_mode == "cloud"


def test_cloud_settings_default_to_safe_values():
    s = Settings(_env_file=None)
    assert s.supabase_url is None
    assert s.azure_openai_api_key is None
    assert s.otel_enabled is False
    assert s.langfuse_host.startswith("https://")
