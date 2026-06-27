"""Generate a coding exercise grounded in retrieved book passages (LLM)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from raggym.practice.models import Exercise

log = get_logger(__name__)

_GEN_PROMPT = """You are RAGGym's exercise author. Create ONE self-contained Python coding \
exercise that teaches this concept: "{topic}".

Ground it in the following passages from the source book — use their terminology and ideas:
---
{context}
---

Requirements:
- Solvable in a single file by implementing one function (pick a clear name and put it in the \
`function_name` field).
- `starter_code`: imports + the function definition with a docstring and a `raise \
NotImplementedError` body. The module MUST import cleanly.
- `test_code`: a pytest file that does `from solution import <function_name>` and has 2-3 test \
functions with concrete asserts that pass for a correct implementation.
- `reference_solution`: a correct full implementation using the same function name.
- Standard library only — no third-party imports.

{format_instructions}
"""


def _format_context(docs: list[Any]) -> tuple[str, list[dict]]:
    lines, sources = [], []
    for i, d in enumerate(docs, start=1):
        m = d.metadata or {}
        tag = f"{m.get('book', '?')} p.{m.get('page', '?')}"
        if m.get("section"):
            tag += f" §{m['section']}"
        lines.append(f"[{i}] ({tag})\n{d.page_content}")
        sources.append(
            {"n": i, "tag": tag, "book": m.get("book"), "page": m.get("page"),
             "section": m.get("section")}
        )
    return "\n\n".join(lines), sources


def generate_exercise(
    topic: str,
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    retriever: Any | None = None,
) -> tuple[Exercise, list[dict]]:
    """Retrieve context for ``topic`` and have the LLM author an exercise."""
    settings = settings or get_settings()

    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from raggym.practice.models import Exercise

    owns_retriever = False
    if retriever is None:
        from raggym.retrieval import RagRetriever

        retriever = RagRetriever(settings, llm=llm)
        owns_retriever = True
    try:
        docs = retriever.retrieve(topic)
    finally:
        if owns_retriever:
            retriever.close()
    context, sources = _format_context(docs)

    if llm is None:
        from raggym.llm import get_llm

        llm = get_llm(settings)

    parser = PydanticOutputParser(pydantic_object=Exercise)
    prompt = ChatPromptTemplate.from_template(_GEN_PROMPT).partial(
        format_instructions=parser.get_format_instructions()
    )
    exercise = (prompt | llm | parser).invoke(
        {"topic": topic, "context": context or "(no context retrieved)"}
    )
    log.info("exercise_generated", topic=topic, function=exercise.function_name)
    return exercise, sources
