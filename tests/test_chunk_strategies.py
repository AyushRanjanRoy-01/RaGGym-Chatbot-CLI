"""Tests for chunking-strategy dispatch + strategy tagging."""

import pytest

pytest.importorskip("langchain_text_splitters")

from raggym.ingestion.chunkers import chunk_pages_by_strategy  # noqa: E402
from raggym.ingestion.parsers import ParsedPage  # noqa: E402


def _pages(text: str):
    return [ParsedPage(page=1, text=text)]


def test_recursive_tags_strategy():
    docs = chunk_pages_by_strategy(
        _pages("Some content about agents and retrieval systems here."),
        strategy="recursive",
        book="b",
        source="b.md",
    )
    assert docs
    assert all(d.metadata["strategy"] == "recursive" for d in docs)


def test_fixed_size_and_tag():
    docs = chunk_pages_by_strategy(
        _pages("word " * 400),
        strategy="fixed",
        book="b",
        source="b.md",
        chunk_size=200,
        chunk_overlap=20,
    )
    assert len(docs) > 1
    assert all(d.metadata["strategy"] == "fixed" for d in docs)
    assert all(len(d.page_content) <= 220 for d in docs)


def test_semantic_falls_back_when_unavailable():
    docs = chunk_pages_by_strategy(
        _pages("Readable content about tool use and multi-agent systems here."),
        strategy="semantic",
        book="b",
        source="b.md",
        embeddings=None,
    )
    assert docs
    assert all(d.metadata["strategy"] == "recursive" for d in docs)
