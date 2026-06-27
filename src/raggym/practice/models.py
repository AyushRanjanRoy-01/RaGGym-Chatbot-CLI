"""Data model for a generated practice exercise."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Exercise(BaseModel):
    """A self-contained coding exercise grounded in the book corpus."""

    title: str = Field(description="Short exercise title.")
    concept: str = Field(description="The RAG/agent concept being practiced.")
    difficulty: str = Field(description="easy | medium | hard")
    statement: str = Field(description="Markdown problem statement explaining the task.")
    function_name: str = Field(description="The function the learner must implement.")
    starter_code: str = Field(
        description="Python starter file defining the function stub (raises NotImplementedError) "
        "with any needed imports. The learner edits this."
    )
    reference_solution: str = Field(description="A correct reference implementation (Python).")
    test_code: str = Field(
        description="pytest file that does `from solution import <function_name>` and asserts "
        "behaviour with 2-3 test functions."
    )
