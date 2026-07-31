"""Fixed-size chunking — a simple character-window baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document

    from raggym.ingestion.parsers import ParsedPage


def fixed_size_chunks(
    pages: list[ParsedPage],
    *,
    book: str,
    source: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split each page into fixed-size character windows (with overlap)."""
    from langchain_core.documents import Document
    from langchain_text_splitters import CharacterTextSplitter

    splitter = CharacterTextSplitter(
        separator="", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs: list[Document] = []
    for page in pages:
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if piece:
                docs.append(
                    Document(
                        page_content=piece,
                        metadata={"book": book, "source": source, "page": page.page},
                    )
                )
    return docs
