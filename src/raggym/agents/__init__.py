"""LangGraph multi-agent orchestration.

chat_graph [Phase 2]
    retrieve → (grade → transform → retrieve)* → generate (cited answer)

practice_graph [Phase 3]
    tutor (pick concept + objective) → exercise generator (grounded in the book)
    → reviewer/grader (evaluate the learner's code/answer with references)
"""

from raggym.agents.chat_graph import answer, build_chat_graph

__all__ = ["answer", "build_chat_graph"]
