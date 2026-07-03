"""Dedicated retrieval tests: weighted RRF, BM25 sparse, and rerank."""

from types import SimpleNamespace

import pytest

pytest.importorskip("rank_bm25")
pytest.importorskip("langchain_core")

from langchain_core.documents import Document  # noqa: E402

from raggym.config import Settings  # noqa: E402
from raggym.retrieval.engine import RagRetriever, _doc_key, _rrf_fuse  # noqa: E402
from raggym.retrieval.rerank import rerank  # noqa: E402


def _doc(text: str, page: int) -> Document:
    return Document(page_content=text, metadata={"source": "s", "page": page})


# ── weighted RRF ─────────────────────────────────────────────────────────────
def test_rrf_weights_break_ties_toward_heavier_list():
    a, b = _doc("alpha", 1), _doc("beta", 2)
    # dense ranks a>b (weight .7), sparse ranks b>a (weight .3) → a should win.
    fused = _rrf_fuse([(0.7, [a, b]), (0.3, [b, a])], k=60)
    assert fused[0].page_content == "alpha"


def test_rrf_dedupes_same_document():
    a = _doc("alpha", 1)
    fused = _rrf_fuse([(1.0, [a]), (1.0, [a])], k=60)
    assert len(fused) == 1


def test_rrf_agreement_outranks_single_list_hit():
    a, b = _doc("alpha", 1), _doc("beta", 2)
    # 'a' appears in both lists (agreement), 'b' only in one → a ranks first.
    fused = _rrf_fuse([(0.5, [a]), (0.5, [b, a])], k=60)
    assert fused[0].page_content == "alpha"


def test_doc_key_distinguishes_pages():
    assert _doc_key(_doc("x", 1)) != _doc_key(_doc("x", 2))


# ── rerank graceful degradation ──────────────────────────────────────────────
def test_rerank_returns_top_n_when_flashrank_missing():
    docs = [_doc(f"c{i}", i) for i in range(6)]
    out = rerank("query", docs, model="ms-marco-MiniLM-L-12-v2", top_n=3)
    assert len(out) == 3


def test_rerank_empty_input():
    assert rerank("q", [], model="m", top_n=3) == []


# ── BM25 sparse retrieval ────────────────────────────────────────────────────
class _FakeQdrant:
    collection_name = "raggym"

    def __init__(self, payloads):
        self.client = SimpleNamespace(
            scroll=lambda **_: ([SimpleNamespace(payload=p) for p in payloads], None)
        )

    def similarity_search(self, query, k=4):
        return []


def test_bm25_ranks_exact_term_first():
    # A discriminative corpus: "BM25"/"ranking" appear in only one of several docs.
    payloads = [
        {"page_content": "Reflection lets an agent critique its own output.",
         "metadata": {"source": "s", "page": 1}},
        {"page_content": "Prompt chaining splits a task into sequential steps.",
         "metadata": {"source": "s", "page": 2}},
        {"page_content": "Tool use lets an agent call external functions.",
         "metadata": {"source": "s", "page": 3}},
        {"page_content": "BM25 is a lexical ranking algorithm using term frequency.",
         "metadata": {"source": "s", "page": 4}},
    ]
    r = RagRetriever(
        Settings(_env_file=None, vector_store="qdrant"),
        vectorstore=_FakeQdrant(payloads),
    )
    out = r._keyword_search("what is BM25 ranking", limit=2)
    assert out
    assert "BM25" in out[0].page_content
