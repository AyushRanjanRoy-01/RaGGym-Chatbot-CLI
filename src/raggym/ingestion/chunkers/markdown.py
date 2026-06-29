"""Markdown-aware chunking.

Splits each page's Markdown with a recursive splitter that *prefers* breaking on
headings and paragraph boundaries, so code blocks and prose stay coherent. Page
numbers are preserved per chunk (for citations) and the nearest Markdown heading
is captured as ``section``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from raggym.ingestion.parsers import ParsedPage

log = get_logger(__name__)

# Prefer splitting at headings, then blank lines, then lines, then spaces.
_SEPARATORS = ["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""]
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_WORD_RE = re.compile(r"[A-Za-z0-9]{2,}")


def _nearest_heading(text: str) -> str:
    matches = _HEADING_RE.findall(text)
    return matches[-1].strip() if matches else ""


def _has_searchable_text(text: str) -> bool:
    """Skip Markdown artifacts such as bare code fences that pollute retrieval."""

    stripped = text.strip()
    if len(stripped) < 20:
        return False
    return len(_WORD_RE.findall(stripped)) >= 3


def chunk_pages(
    pages: list[ParsedPage],
    *,
    book: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Chunk parsed pages into LangChain ``Document`` objects with metadata.

    Metadata per chunk: ``book``, ``source`` (filename), ``page`` (1-based),
    ``section`` (nearest heading), ``chunk`` (per-page index).
    """
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        keep_separator=True,
    )

    docs: list[Document] = []
    for page in pages:
        pieces = splitter.split_text(page.text)
        for idx, piece in enumerate(pieces):
            piece = piece.strip()
            if not _has_searchable_text(piece):
                continue
            docs.append(
                Document(
                    page_content=piece,
                    metadata={
                        "book": book,
                        "source": source,
                        "page": page.page,
                        "section": _nearest_heading(piece),
                        "chunk": idx,
                    },
                )
            )

    log.info("chunk_pages_done", book=book, pages=len(pages), chunks=len(docs))
    return docs
