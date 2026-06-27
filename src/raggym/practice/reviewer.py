"""AI review of a learner's solution against the reference + book concept (LLM)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from raggym.config import Settings, get_settings
from raggym.core import get_logger

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

log = get_logger(__name__)

_REVIEW_PROMPT = """You are a senior engineer reviewing a learner's solution to a RAG/agent \
coding exercise on the concept: "{concept}".

Learner's solution:
```python
{solution}
```

Reference solution:
```python
{reference}
```

Give concise, encouraging feedback in markdown with these sections:
- **Correctness** — does it work / match the reference's behaviour?
- **Alignment with the concept** — how well does it reflect "{concept}" as taught in the book?
- **Suggestions** — concrete improvements.
- **Score** — X/5.
"""


def review(
    exercise_dir: str | Path,
    *,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
) -> str:
    settings = settings or get_settings()
    base = Path(exercise_dir)
    meta = json.loads((base / "meta.json").read_text(encoding="utf-8"))
    solution = (base / "solution.py").read_text(encoding="utf-8")
    ref_file = base / "_reference.py"
    reference = ref_file.read_text(encoding="utf-8") if ref_file.exists() else "(unavailable)"

    if llm is None:
        from raggym.llm import get_llm

        llm = get_llm(settings)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    chain = ChatPromptTemplate.from_template(_REVIEW_PROMPT) | llm | StrOutputParser()
    feedback = chain.invoke(
        {"concept": meta.get("concept", "?"), "solution": solution, "reference": reference}
    )
    log.info("review_done", dir=str(base))
    return feedback
