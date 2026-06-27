"""Unit tests for markdown-aware chunking (no network / model needed)."""

import pytest

# Skip this module entirely if the splitter dependency isn't installed
# (keeps the lean base/dev CI green).
pytest.importorskip("langchain_text_splitters")

from raggym.ingestion.chunkers import chunk_pages  # noqa: E402
from raggym.ingestion.parsers import ParsedPage  # noqa: E402

_PAGE = ParsedPage(
    page=1,
    text=(
        "# Chapter 1: Agents\n\n"
        "An agent perceives and acts.\n\n"
        "## Tool Use\n\n"
        "```python\n"
        "def call_tool(name):\n"
        "    return registry[name]()\n"
        "```\n"
    ),
)


def test_metadata_attached():
    docs = chunk_pages([_PAGE], book="ai", source="ai.pdf", chunk_size=200, chunk_overlap=20)
    assert docs
    for d in docs:
        assert d.metadata["book"] == "ai"
        assert d.metadata["source"] == "ai.pdf"
        assert d.metadata["page"] == 1
        assert "section" in d.metadata
        assert "chunk" in d.metadata


def test_nearest_heading_captured():
    docs = chunk_pages([_PAGE], book="ai", source="ai.pdf", chunk_size=2000, chunk_overlap=0)
    sections = {d.metadata["section"] for d in docs}
    assert any("Chapter 1" in s or "Tool Use" in s for s in sections)


def test_code_preserved():
    docs = chunk_pages([_PAGE], book="ai", source="ai.pdf", chunk_size=2000, chunk_overlap=0)
    joined = "\n".join(d.page_content for d in docs)
    assert "def call_tool(name):" in joined


def test_respects_chunk_size():
    long_page = ParsedPage(page=2, text="sentence number text. " * 400)
    docs = chunk_pages([long_page], book="ai", source="ai.pdf", chunk_size=200, chunk_overlap=20)
    assert len(docs) > 1
    # RecursiveCharacterTextSplitter may slightly exceed size with kept separators.
    assert all(len(d.page_content) <= 300 for d in docs)


def test_empty_pages_dropped():
    docs = chunk_pages([ParsedPage(page=3, text="   \n  ")], book="ai", source="ai.pdf")
    assert docs == []
