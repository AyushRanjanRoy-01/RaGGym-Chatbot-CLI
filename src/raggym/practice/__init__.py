"""Practice harness [Phase 3].

Turns book concepts into hands-on coding practice:
    generator   LLM authors an exercise grounded in retrieved passages
    workspace   materialises it under workspace/<slug>/ (edit solution.py)
    grader      runs the learner's solution against pytest checks (no LLM)
    reviewer    LLM feedback vs. the reference + the book's approach
"""

from raggym.practice.service import create_exercise, grade_exercise, list_exercises

__all__ = ["create_exercise", "grade_exercise", "list_exercises"]
