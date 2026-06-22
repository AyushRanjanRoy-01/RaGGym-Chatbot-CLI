"""Smoke tests for the configuration and logging foundation."""

import pytest

from raggym import __version__
from raggym.config import Settings, get_settings


def test_version_is_set():
    assert __version__
    assert isinstance(__version__, str)


def test_settings_load_with_defaults(monkeypatch, tmp_path):
    # Ignore any developer .env so defaults are deterministic in CI.
    monkeypatch.chdir(tmp_path)
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
