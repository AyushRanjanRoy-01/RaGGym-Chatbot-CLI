"""Streamlit chat UI for RAGGym — grounded Q&A over the book corpus.

Launch with ``raggym chat`` (or ``streamlit run src/raggym/apps/chat_app.py``).
"""

from __future__ import annotations

import streamlit as st

from raggym.agents import answer, build_chat_graph
from raggym.config import get_settings

st.set_page_config(page_title="RAGGym · Chat", page_icon="🏋️", layout="centered")
settings = get_settings()


@st.cache_resource(show_spinner="Loading retriever + model…")
def _graph():
    return build_chat_graph(settings=settings)


with st.sidebar:
    st.header("🏋️ RAGGym")
    st.caption("Grounded Q&A over your RAG book corpus.")
    st.subheader("Config")
    st.write(
        {
            "llm": f"{settings.llm_provider}:{settings.llm_model}",
            "embeddings": f"{settings.embed_provider}:{settings.embed_model}",
            "vector_store": settings.vector_store,
            "hybrid": settings.use_hybrid,
            "multi_query": settings.use_multi_query,
            "reranker": settings.use_reranker,
            "corrective": settings.use_corrective,
            "top_k": settings.retrieval_top_k,
        }
    )
    if settings.llm_provider == "ollama":
        st.info("Using local Ollama. Ensure `ollama serve` is running and the model is pulled.")

st.title("Chat with the corpus")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"**[{s['n']}]** {s['tag']}")

if question := st.chat_input("Ask about RAG, agents, retrieval…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving + reasoning…"):
                result = answer(_graph(), question)
            st.markdown(result["generation"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.markdown(f"**[{s['n']}]** {s['tag']}")
            st.session_state.messages.append(
                {"role": "assistant", "content": result["generation"], "sources": result["sources"]}
            )
        except Exception as exc:  # noqa: BLE001 — surface provider/runtime errors in-UI
            st.error(
                f"Generation failed: {exc}\n\n"
                "Check that your LLM provider is configured (Ollama running, or an "
                "API key set in `.env`)."
            )
