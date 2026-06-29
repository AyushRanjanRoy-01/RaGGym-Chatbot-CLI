"""LangGraph chat agent over the book corpus.

Default flow:        retrieve → generate (cited answer)
Corrective flow:     retrieve → grade → [transform → retrieve]* → generate
(enabled with USE_CORRECTIVE; retries capped at MAX_RETRIES)

The builder takes optional ``llm`` and ``retriever`` for dependency injection,
so the graph can be unit-tested with fakes and no live model.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, TypedDict

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)

_SYSTEM = (
    "You are RAGGym, a tutor that helps people learn Retrieval-Augmented Generation. "
    "Answer the question using ONLY the numbered context passages. Cite the passages "
    "you use inline as [n]. Be precise and concise, and prefer the source's own "
    "terminology. If the answer is not in the context, say you don't have enough "
    "information in the corpus to answer — do not invent facts."
)

_GREETING_TERMS = {"hello", "hey", "hi", "hola", "yo"}
_THANKS_TERMS = {"thank", "thanks", "ty"}
_HELP_QUERIES = {
    "help",
    "what can you do",
    "who are you",
}


class ChatState(TypedDict, total=False):
    question: str
    original_question: str
    documents: list[Any]
    sources: list[dict]
    generation: str
    retries: int


def _format_context(docs: list[Document]) -> tuple[str, list[dict]]:
    lines, sources = [], []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        tag = f"{m.get('book', '?')} p.{m.get('page', '?')}"
        if m.get("section"):
            tag += f" §{m['section']}"
        lines.append(f"[{i}] ({tag})\n{d.page_content}")
        sources.append(
            {
                "n": i,
                "tag": tag,
                "book": m.get("book"),
                "page": m.get("page"),
                "section": m.get("section"),
                "snippet": d.page_content[:500],
            }
        )
    return "\n\n".join(lines), sources


def _parse_indices(text: str, n: int) -> list[int]:
    if "NONE" in text.upper():
        return []
    return sorted({int(x) for x in re.findall(r"\d+", text) if 0 <= int(x) < n})


def _small_talk_answer(question: str) -> str | None:
    """Answer simple non-corpus turns without paying retrieval/LLM latency."""

    normalized = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    words = [word for word in normalized.split() if word]
    phrase = " ".join(words)
    if not words:
        return None

    if phrase in _HELP_QUERIES:
        return (
            "I can answer questions grounded in your uploaded PDFs. Upload a PDF in the "
            "sidebar, click `Save + ingest`, then ask about the document."
        )

    if len(words) <= 3 and any(word in _GREETING_TERMS for word in words):
        return (
            "Hey, I am ready. Upload or ingest a PDF, then ask me a question about "
            "the corpus."
        )

    if len(words) <= 4 and any(word in _THANKS_TERMS for word in words):
        return "You got it. Ask me anything from the uploaded corpus."

    return None


def build_chat_graph(
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    retriever: Any | None = None,
):
    """Compile and return the chat graph (a LangGraph ``CompiledStateGraph``)."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langgraph.graph import END, START, StateGraph

    settings = settings or get_settings()
    if llm is None:
        from raggym.llm import get_llm

        llm = get_llm(settings)
    if retriever is None:
        from raggym.retrieval import RagRetriever

        retriever = RagRetriever(settings, llm=llm)

    answer_chain = (
        ChatPromptTemplate.from_messages(
            [("system", _SYSTEM), ("human", "Question: {question}\n\nContext:\n{context}")]
        )
        | llm
        | StrOutputParser()
    )
    grade_chain = (
        ChatPromptTemplate.from_template(
            "Question: {q}\n\nPassages:\n{listing}\n\n"
            "Return the numbers of the passages RELEVANT to answering the question, "
            "comma-separated (e.g. 0,2). If none are relevant, return NONE."
        )
        | llm
        | StrOutputParser()
    )
    rewrite_chain = (
        ChatPromptTemplate.from_template(
            "Rewrite the question to improve document retrieval (more specific keywords "
            "and synonyms). Return only the rewritten question.\n\nQuestion: {q}"
        )
        | llm
        | StrOutputParser()
    )

    def retrieve(state: ChatState) -> dict:
        return {"documents": retriever.retrieve(state["question"])}

    def grade(state: ChatState) -> dict:
        docs = state["documents"]
        if not docs:
            return {"documents": []}
        listing = "\n".join(f"{i}. {d.page_content[:200]}" for i, d in enumerate(docs))
        out = grade_chain.invoke({"q": state["question"], "listing": listing})
        keep = _parse_indices(out, len(docs))
        return {"documents": [docs[i] for i in keep]}

    def transform(state: ChatState) -> dict:
        new_q = rewrite_chain.invoke({"q": state["question"]}).strip()
        log.info("query_transformed", new_question=new_q)
        return {"question": new_q, "retries": state.get("retries", 0) + 1}

    def generate(state: ChatState) -> dict:
        docs = state.get("documents") or []
        context, sources = _format_context(docs)
        if not docs:
            context = "(no relevant passages found)"
        question = state.get("original_question") or state["question"]
        answer = answer_chain.invoke({"question": question, "context": context})
        return {"generation": answer, "sources": sources, "documents": docs}

    def route_after_grade(state: ChatState) -> str:
        if state["documents"] or state.get("retries", 0) >= settings.max_retries:
            return "generate"
        return "transform"

    g = StateGraph(ChatState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_edge(START, "retrieve")
    if settings.use_corrective:
        g.add_node("grade", grade)
        g.add_node("transform", transform)
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges(
            "grade", route_after_grade, {"generate": "generate", "transform": "transform"}
        )
        g.add_edge("transform", "retrieve")
    else:
        g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()


def answer(graph, question: str) -> dict:
    """Run the graph for a single question; return {generation, sources, documents}."""
    if response := _small_talk_answer(question):
        return {"generation": response, "sources": [], "documents": []}

    state = graph.invoke(
        {"question": question, "original_question": question, "retries": 0}
    )
    return {
        "generation": state.get("generation", ""),
        "sources": state.get("sources", []),
        "documents": state.get("documents", []),
    }
