"""Read/write practice exercises on disk under ``workspace/``.

Each exercise is a flat directory so pytest's default import mode puts it on
``sys.path`` (``from solution import ...`` just works)::

    workspace/<slug>/
        README.md          problem statement + source citations
        solution.py        the learner edits this (starts as a stub)
        test_exercise.py    pytest checks
        _reference.py      hidden reference solution
        meta.json          concept, difficulty, function_name, sources
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from raggym.practice.models import Exercise

log = get_logger(__name__)


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s or "exercise"


def _readme(exercise: Exercise, sources: list[dict]) -> str:
    cites = "\n".join(f"- {s['tag']}" for s in sources) or "- (no sources)"
    return (
        f"# {exercise.title}\n\n"
        f"**Concept:** {exercise.concept}  \n**Difficulty:** {exercise.difficulty}\n\n"
        f"{exercise.statement}\n\n"
        f"## Your task\n\nImplement `{exercise.function_name}` in `solution.py`, then run:\n\n"
        f"```bash\nraggym practice grade {{this_directory}}\n```\n\n"
        f"## Based on\n\n{cites}\n"
    )


def write_exercise(
    exercise: Exercise, sources: list[dict], *, settings: Settings | None = None
) -> Path:
    """Materialise an exercise on disk; return its directory."""
    settings = settings or get_settings()
    base = settings.workspace_dir / _slug(exercise.title)
    base.mkdir(parents=True, exist_ok=True)

    (base / "README.md").write_text(_readme(exercise, sources), encoding="utf-8")
    (base / "solution.py").write_text(exercise.starter_code.rstrip() + "\n", encoding="utf-8")
    (base / "test_exercise.py").write_text(exercise.test_code.rstrip() + "\n", encoding="utf-8")
    (base / "_reference.py").write_text(
        exercise.reference_solution.rstrip() + "\n", encoding="utf-8"
    )
    (base / "meta.json").write_text(
        json.dumps(
            {
                "title": exercise.title,
                "concept": exercise.concept,
                "difficulty": exercise.difficulty,
                "function_name": exercise.function_name,
                "sources": sources,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("exercise_written", path=str(base))
    return base


def list_exercises(settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    root = settings.workspace_dir
    if not root.exists():
        return []
    out = []
    for meta_file in sorted(root.glob("*/meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out.append({"dir": str(meta_file.parent), **meta})
    return out
