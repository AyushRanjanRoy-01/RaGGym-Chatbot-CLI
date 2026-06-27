"""RAGAS evaluation of the RAG chat pipeline over a question set.

Runs the chat graph for each question, collects the answer + retrieved contexts,
then scores them with RAGAS metrics (faithfulness, answer relevancy, context
precision, context recall). Needs an LLM provider and the ``eval`` extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)

_DEFAULT_QUESTIONS = Path(__file__).parent / "questions.json"


def load_questions(path: str | Path | None = None) -> list[dict]:
    path = Path(path) if path else _DEFAULT_QUESTIONS
    return json.loads(path.read_text(encoding="utf-8"))


def build_samples(
    questions: list[dict],
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
) -> list[dict]:
    """Run the chat graph per question → RAGAS SingleTurnSample-shaped rows."""
    settings = settings or get_settings()
    from raggym.agents import answer as run_answer
    from raggym.agents import build_chat_graph
    from raggym.retrieval import RagRetriever

    if llm is None:
        from raggym.llm import get_llm

        llm = get_llm(settings)

    retriever = RagRetriever(settings, llm=llm)
    graph = build_chat_graph(settings=settings, llm=llm, retriever=retriever)
    rows: list[dict] = []
    try:
        for item in questions:
            result = run_answer(graph, item["question"])
            rows.append(
                {
                    "user_input": item["question"],
                    "response": result["generation"],
                    "retrieved_contexts": [d.page_content for d in result["documents"]],
                    "reference": item.get("ground_truth", ""),
                }
            )
            log.info("eval_sample_built", question=item["question"][:60])
    finally:
        retriever.close()
    return rows


def evaluate_pipeline(
    *,
    questions_path: str | Path | None = None,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    embeddings: Embeddings | None = None,
) -> Any:
    """Evaluate the pipeline with RAGAS; return the RAGAS result object."""
    settings = settings or get_settings()

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS is not importable in this environment. Install the eval extra "
            '(`pip install -e ".[eval]"`). Note: ragas can lag the langchain 1.x '
            "stack (it may import a langchain-community module that was removed), so a "
            f"compatible pin set may be required. Original error: {exc}"
        ) from exc

    if llm is None:
        from raggym.llm import get_llm

        llm = get_llm(settings)
    if embeddings is None:
        from raggym.embeddings import get_embeddings

        embeddings = get_embeddings(settings)

    rows = build_samples(load_questions(questions_path), settings=settings, llm=llm)
    dataset = EvaluationDataset.from_list(rows)

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
    )
    log.info("eval_complete")
    return result
