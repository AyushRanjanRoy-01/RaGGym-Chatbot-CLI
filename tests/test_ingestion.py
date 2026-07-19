"""Tests for multi-format document loaders."""

import pytest

pytest.importorskip("pymupdf4llm")

from raggym.ingestion.parsers import load_document  # noqa: E402
from raggym.ingestion.parsers.text import parse_html, parse_text_file  # noqa: E402


def test_parse_text_file(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# Title\n\nBody text here.", encoding="utf-8")
    pages = parse_text_file(p)
    assert len(pages) == 1
    assert "Body text here." in pages[0].text


def test_parse_html_strips_scripts_and_styles(tmp_path):
    p = tmp_path / "a.html"
    p.write_text(
        "<html><head><style>x{color:red}</style></head>"
        "<body><h1>Hi</h1><script>bad()</script><p>Real content</p></body></html>",
        encoding="utf-8",
    )
    text = parse_html(p)[0].text
    assert "Real content" in text
    assert "bad()" not in text
    assert "color:red" not in text


def test_load_document_dispatch(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("plain text file", encoding="utf-8")
    assert load_document(p)[0].text == "plain text file"


def test_load_document_rejects_unknown_type(tmp_path):
    p = tmp_path / "a.xyz"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(p)
