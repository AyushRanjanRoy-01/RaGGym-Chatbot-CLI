"""Tests for the raw-document parse cache."""

import pytest

pytest.importorskip("pymupdf4llm")

from raggym.ingestion.cache import load_pages_cached  # noqa: E402
from raggym.ingestion.parsers import ParsedPage  # noqa: E402


def _loader_factory(calls):
    def loader(path, *, max_pages=None):
        calls["n"] += 1
        return [ParsedPage(page=1, text="parsed")]

    return loader


def test_cache_miss_then_hit(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("hello world", encoding="utf-8")
    cache = tmp_path / "cache"
    calls = {"n": 0}
    loader = _loader_factory(calls)

    load_pages_cached(f, cache_dir=cache, loader=loader)
    second = load_pages_cached(f, cache_dir=cache, loader=loader)

    assert calls["n"] == 1  # second call served from cache
    assert second[0].text == "parsed"
    assert second[0].page == 1


def test_cache_invalidates_on_content_change(tmp_path):
    f = tmp_path / "doc.txt"
    f.write_text("v1", encoding="utf-8")
    cache = tmp_path / "cache"
    calls = {"n": 0}
    loader = _loader_factory(calls)

    load_pages_cached(f, cache_dir=cache, loader=loader)
    f.write_text("v2 changed content", encoding="utf-8")
    load_pages_cached(f, cache_dir=cache, loader=loader)

    assert calls["n"] == 2  # different hash → re-parsed
