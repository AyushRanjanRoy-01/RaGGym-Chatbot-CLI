"""High-level practice operations used by the CLI."""

from __future__ import annotations

from pathlib import Path

from raggym.config import Settings, get_settings
from raggym.core import get_logger
from raggym.practice.grader import run_tests
from raggym.practice.workspace import list_exercises, write_exercise

log = get_logger(__name__)


def create_exercise(topic: str, *, settings: Settings | None = None) -> Path:
    """Generate an exercise for ``topic`` and write it to the workspace."""
    settings = settings or get_settings()
    from raggym.practice.generator import generate_exercise

    exercise, sources = generate_exercise(topic, settings=settings)
    return write_exercise(exercise, sources, settings=settings)


def grade_exercise(exercise_dir: str | Path, *, settings: Settings | None = None) -> dict:
    """Run the exercise tests, then ask the LLM to review the solution.

    Test running never needs an LLM; the review degrades gracefully if no
    provider is reachable.
    """
    settings = settings or get_settings()
    tests = run_tests(exercise_dir)
    try:
        from raggym.practice.reviewer import review

        feedback = review(exercise_dir, settings=settings)
    except Exception as exc:  # noqa: BLE001 — review is best-effort
        feedback = (
            f"_AI review unavailable: {exc}._\n\n"
            "Set an LLM provider (run Ollama, or set an API key in `.env`) to enable review."
        )
    return {"tests": tests, "review": feedback}


__all__ = ["create_exercise", "grade_exercise", "list_exercises"]
