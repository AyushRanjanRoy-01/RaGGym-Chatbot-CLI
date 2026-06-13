from abc import ABC, abstractmethod

import structlog
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

log = structlog.get_logger()


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]: ...


class RecursiveChunker(BaseChunker):
    """Splits on paragraph → line → word → char boundaries in that order."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size or settings.chunk_size,
            chunk_overlap=chunk_overlap or settings.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        log.info("chunking_complete", input_docs=len(documents), output_chunks=len(chunks))
        return chunks
