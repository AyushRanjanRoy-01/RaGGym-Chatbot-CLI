"""
RAGAS evaluation for the RAG pipeline.

Usage:
    python -m eval.evaluate --questions eval/sample_questions.json
    python -m eval.evaluate --questions eval/sample_questions.json --ground-truths eval/ground_truths.json
"""

import argparse
import json

import structlog
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from core.logging import setup_logging
from rag.chain import build_rag_chain
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore

setup_logging()
log = structlog.get_logger()


def run_evaluation(questions: list[str], ground_truths: list[str] | None = None) -> dict:
    embeddings = get_embeddings()
    store = get_vectorstore(embeddings)
    retriever = store.as_retriever()
    chain = build_rag_chain(retriever)

    answers, contexts = [], []

    for question in questions:
        docs = retriever.invoke(question)
        contexts.append([doc.page_content for doc in docs])
        answers.append(chain.invoke({"question": question, "history": []}))
        log.info("eval_question_done", question=question[:60])

    data: dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    metrics = [faithfulness, answer_relevancy, context_precision]

    if ground_truths:
        data["ground_truth"] = ground_truths
        metrics.append(context_recall)

    result = evaluate(Dataset.from_dict(data), metrics=metrics)
    log.info("eval_complete", scores=str(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline using RAGAS metrics")
    parser.add_argument("--questions", required=True, help="JSON file containing a list of question strings")
    parser.add_argument("--ground-truths", help="JSON file containing a list of ground truth answer strings")
    args = parser.parse_args()

    with open(args.questions) as f:
        questions: list[str] = json.load(f)

    ground_truths = None
    if args.ground_truths:
        with open(args.ground_truths) as f:
            ground_truths = json.load(f)

    results = run_evaluation(questions, ground_truths)

    print("\n=== RAGAS Evaluation Results ===")
    print(results)


if __name__ == "__main__":
    main()
