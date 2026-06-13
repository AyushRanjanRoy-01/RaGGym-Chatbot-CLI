from abc import ABC, abstractmethod
from pathlib import Path

import structlog
from langchain_community.document_loaders import TextLoader as _LCTextLoader
from langchain_core.documents import Document

log = structlog.get_logger()


class BaseLoader(ABC):
    @abstractmethod
    def load(self, path: str) -> list[Document]: ...


class TextLoader(BaseLoader):
    """Loads all .txt files recursively from a directory, or a single .txt file."""

    def load(self, path: str) -> list[Document]:
        p = Path(path)
        files = sorted(p.rglob("*.txt")) if p.is_dir() else [p]

        if not files:
            raise FileNotFoundError(f"No .txt files found at: {path}")

        docs: list[Document] = []
        for f in files:
            loaded = _LCTextLoader(str(f), encoding="utf-8").load()
            for doc in loaded:
                doc.metadata["source"] = f.name
            docs.extend(loaded)
            log.info("file_loaded", file=f.name, pages=len(loaded))

        log.info("load_complete", total_docs=len(docs), path=str(path))
        return docs
