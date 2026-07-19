"""Loaders for text formats: Markdown, plain text, and HTML."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from raggym.ingestion.parsers.pdf import ParsedPage


def parse_text_file(path: str | Path) -> list[ParsedPage]:
    """Load a Markdown or plain-text file as a single page."""
    text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    return [ParsedPage(page=1, text=text)] if text else []


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self._parts.append(data.strip())

    @property
    def text(self) -> str:
        return "\n".join(self._parts)


def parse_html(path: str | Path) -> list[ParsedPage]:
    """Load an HTML file, stripped to readable text (script/style removed)."""
    extractor = _TextExtractor()
    extractor.feed(Path(path).read_text(encoding="utf-8", errors="ignore"))
    text = re.sub(r"\n{3,}", "\n\n", extractor.text).strip()
    return [ParsedPage(page=1, text=text)] if text else []
