"""Optional cross-encoder reranking via the lightweight ``flashrank`` library.

Flashrank runs small ONNX cross-encoders (no torch). It's an optional extra; if
not installed, reranking degrades gracefully to the original top-N ordering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document

log = get_logger(__name__)

_RANKERS: dict = {}


def rerank(query: str, docs: list[Document], *, model: str, top_n: int) -> list[Document]:
    """Rerank ``docs`` against ``query`` with a cross-encoder; return top-N."""
    if not docs:
        return docs
    try:
        from flashrank import Ranker, RerankRequest
    except ImportError:
        log.warning("reranker_unavailable", hint="pip install 'raggym[rerank]'")
        return docs[:top_n]

    ranker = _RANKERS.get(model)
    if ranker is None:
        ranker = Ranker(model_name=model)
        _RANKERS[model] = ranker

    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    ranked = ranker.rerank(RerankRequest(query=query, passages=passages))

    out: list[Document] = []
    for r in ranked[:top_n]:
        doc = docs[int(r["id"])]
        doc.metadata["rerank_score"] = float(r["score"])
        out.append(doc)
    log.info("rerank_done", candidates=len(docs), kept=len(out))
    return out
