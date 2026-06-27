"""Evaluation [Phase 4].

RAGAS metrics — faithfulness, answer relevancy, context precision, context
recall — to measure retrieval/answer quality as you tune the pipeline.
Requires the ``eval`` extra and an LLM provider.
"""

from raggym.eval.runner import build_samples, evaluate_pipeline, load_questions

__all__ = ["build_samples", "evaluate_pipeline", "load_questions"]
