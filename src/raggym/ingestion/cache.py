"""Raw-document parse cache: skip re-parsing unchanged files on re-ingest.

Parsed pages are cached as JSON keyed by the file's content hash, so a re-ingest
re-chunks cached raw text instead of re-parsing the PDF from scratch.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from raggym.core import get_logger
from raggym.ingestion.parsers import ParsedPage

log = get_logger(__name__)

Loader = Callable[..., list[ParsedPage]]


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            h.update(block)
    return h.hexdigest()[:16]


def load_pages_cached(
    path: str | Path,
    *,
    cache_dir: str | Path,
    loader: Loader,
    max_pages: int | None = None,
) -> list[ParsedPage]:
    """Return parsed pages from cache, else parse via ``loader`` and cache them."""
    path = Path(path)
    cache_dir = Path(cache_dir)
    key = f"{_file_hash(path)}.p{max_pages if max_pages else 'all'}"
    cache_file = cache_dir / f"{path.name}.{key}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        log.info("parse_cache_hit", file=path.name)
        return [ParsedPage(page=d["page"], text=d["text"]) for d in data]

    pages = loader(path, max_pages=max_pages)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps([{"page": p.page, "text": p.text} for p in pages]), encoding="utf-8"
    )
    log.info("parse_cache_miss", file=path.name, pages=len(pages))
    return pages
