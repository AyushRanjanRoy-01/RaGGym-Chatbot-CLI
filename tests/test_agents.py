"""Tests for the retrieval engine and chat graph using fakes (no live model)."""

from types import SimpleNamespace

import pytest

pytest.importorskip("langgraph")

from langchain_core.documents import Document  # noqa: E402
from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: E402

from raggym.agents import answer, build_chat_graph  # noqa: E402
from raggym.config import Settings  # noqa: E402
from raggym.retrieval import RagRetriever  # noqa: E402


class _FakeVectorStore:
    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, query, k=4):
        return self._docs[:k]


class _FakeQdrantVectorStore(_FakeVectorStore):
    collection_name = "raggym"

    def __init__(self, docs, payloads):
        super().__init__(docs)
        self.client = SimpleNamespace(
            scroll=lambda **_: ([SimpleNamespace(payload=payload) for payload in payloads], None)
        )


class _FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def retrieve(self, query):
        return self._docs


class _ExplodingGraph:
    def invoke(self, state):
        raise AssertionError("small talk should not invoke the graph")


def _docs():
    return [
        Document(
            page_content="Prompt chaining breaks a task into sequential LLM steps.",
            metadata={"book": "AI", "source": "ai.pdf", "page": 42, "section": "Prompt Chaining"},
        )
    ]


def test_retriever_dedup_and_topk():
    docs = [
        Document(page_content=f"chunk {i}", metadata={"source": "s", "page": i}) for i in range(10)
    ]
    settings = Settings(
        _env_file=None, retrieval_top_k=3, use_multi_query=False, use_reranker=False
    )
    retriever = RagRetriever(settings, vectorstore=_FakeVectorStore(docs))
    out = retriever.retrieve("q")
    assert len(out) == 3


def test_retriever_hybrid_prefers_exact_keyword_hits():
    vector_docs = [
        Document(page_content="unrelated agent safety", metadata={"source": "s", "page": 1})
    ]
    payloads = [
        {
            "page_content": "BM25 is a keyword-based retrieval algorithm.",
            "metadata": {"source": "s", "page": 2},
        }
    ]
    settings = Settings(_env_file=None, retrieval_top_k=1, vector_store="qdrant", use_hybrid=True)
    retriever = RagRetriever(settings, vectorstore=_FakeQdrantVectorStore(vector_docs, payloads))

    out = retriever.retrieve("what BM25")

    assert "BM25" in out[0].page_content


def test_chat_graph_generates_with_sources():
    settings = Settings(_env_file=None, use_corrective=False)
    llm = FakeListChatModel(responses=["Prompt chaining is sequential decomposition [1]."])
    graph = build_chat_graph(settings=settings, llm=llm, retriever=_FakeRetriever(_docs()))
    out = answer(graph, "What is prompt chaining?")
    assert "[1]" in out["generation"]
    assert out["sources"][0]["page"] == 42
    assert out["sources"][0]["book"] == "AI"


def test_answer_handles_greeting_without_retrieval():
    out = answer(_ExplodingGraph(), "yo")
    assert "ready" in out["generation"].lower()
    assert out["sources"] == []
    assert out["documents"] == []


def test_corrective_graph_grades_and_generates():
    settings = Settings(_env_file=None, use_corrective=True, max_retries=1)
    # First call grades (keep passage 0), second call generates the answer.
    llm = FakeListChatModel(responses=["0", "Answer grounded in context [1]."])
    graph = build_chat_graph(settings=settings, llm=llm, retriever=_FakeRetriever(_docs()))
    out = answer(graph, "What is prompt chaining?")
    assert out["generation"]
    assert out["documents"]
