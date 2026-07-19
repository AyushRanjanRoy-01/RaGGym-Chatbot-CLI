"""Near-duplicate chunk detection (token-set Jaccard similarity)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dedupe_chunks(docs: list[Document], *, threshold: float = 0.9) -> list[Document]:
    """Drop chunks whose token-set Jaccard vs an already-kept chunk ≥ threshold."""
    kept: list[Document] = []
    kept_sets: list[set[str]] = []
    dropped = 0
    for doc in docs:
        tokens = _token_set(doc.page_content)
        if any(_jaccard(tokens, seen) >= threshold for seen in kept_sets):
            dropped += 1
            continue
        kept.append(doc)
        kept_sets.append(tokens)
    if dropped:
        log.info("dedupe_chunks", dropped=dropped, kept=len(kept))
    return kept
