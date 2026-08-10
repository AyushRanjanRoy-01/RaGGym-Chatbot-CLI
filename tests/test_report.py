"""Tests for the chunking-strategy comparison report."""

import pytest

pytest.importorskip("langchain_text_splitters")

from raggym.ingestion.parsers import ParsedPage  # noqa: E402
from raggym.ingestion.report import compare_chunking_strategies  # noqa: E402


def test_report_returns_stats_per_strategy():
    pages = [ParsedPage(page=1, text="word " * 300)]
    rows = compare_chunking_strategies(
        pages, strategies=("recursive", "fixed"), chunk_size=200, chunk_overlap=20
    )
    assert {r["strategy"] for r in rows} == {"recursive", "fixed"}
    assert all(r["chunks"] > 0 for r in rows)
    assert all({"avg_chars", "min_chars", "max_chars"} <= set(r) for r in rows)
