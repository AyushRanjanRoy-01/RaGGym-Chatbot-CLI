"""Tests for near-duplicate chunk detection."""

from types import SimpleNamespace

import pytest

pytest.importorskip("pymupdf4llm")

from raggym.ingestion.dedup import dedupe_chunks  # noqa: E402


def _doc(text: str):
    return SimpleNamespace(page_content=text, metadata={})


def test_drops_near_duplicates():
    docs = [
        _doc("Reflection lets an agent critique its own output and improve."),
        _doc("Reflection lets an agent critique its own output and improve it."),  # near-dup
        _doc("Prompt chaining splits a task into sequential steps."),
    ]
    out = dedupe_chunks(docs, threshold=0.9)
    assert len(out) == 2


def test_keeps_distinct_chunks():
    docs = [_doc("alpha beta gamma"), _doc("delta epsilon zeta")]
    assert len(dedupe_chunks(docs, threshold=0.9)) == 2


def test_threshold_one_keeps_all_non_identical():
    docs = [_doc("a b c d"), _doc("a b c e")]
    assert len(dedupe_chunks(docs, threshold=1.0)) == 2
