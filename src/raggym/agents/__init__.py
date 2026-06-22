"""LangGraph multi-agent orchestration.

chat_graph [Phase 2]
    router → retrieve → rerank → grade (CRAG-style self-correction) → cite/answer

practice_graph [Phase 3]
    tutor (pick concept + objective) → exercise generator (grounded in the book)
    → reviewer/grader (evaluate the learner's code/answer with references)
"""
