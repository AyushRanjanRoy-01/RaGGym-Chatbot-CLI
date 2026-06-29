"""Streamlit chat UI for RAGGym — grounded Q&A over the book corpus.

Launch with ``raggym chat`` (or ``streamlit run src/raggym/apps/chat_app.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from raggym.agents import answer, build_chat_graph
from raggym.config import get_settings
from raggym.ingestion import ingest_path
from raggym.llm import get_llm
from raggym.retrieval import RagRetriever

st.set_page_config(page_title="RAGGym · Chat", page_icon="🏋️", layout="centered")
settings = get_settings()


def _safe_pdf_name(filename: str) -> str:
    stem = Path(filename).name.strip() or "upload.pdf"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


def _save_upload(uploaded_file) -> Path:
    settings.books_dir.mkdir(parents=True, exist_ok=True)
    target = settings.books_dir / _safe_pdf_name(uploaded_file.name)
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _answer_once(question: str) -> dict:
    """Build per request so local Qdrant is not held open between Streamlit reruns."""

    llm = get_llm(settings)
    retriever = RagRetriever(settings, llm=llm)
    graph = build_chat_graph(settings=settings, llm=llm, retriever=retriever)
    try:
        return answer(graph, question)
    finally:
        retriever.close()


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

    st.subheader("Upload")
    uploaded_pdf = st.file_uploader(
        f"Upload a PDF to `{settings.books_dir}`",
        type=["pdf"],
        accept_multiple_files=False,
    )
    rebuild = st.checkbox(
        "Rebuild vector DB from all saved PDFs",
        value=True,
        help="Recommended after changing embedding provider/model; also avoids duplicate chunks.",
    )
    if uploaded_pdf and st.button("Save + ingest", type="primary"):
        try:
            saved_path = _save_upload(uploaded_pdf)
            with st.spinner("Chunking, embedding, and storing in Qdrant..."):
                ingest_target = settings.books_dir if rebuild else saved_path
                result = ingest_path(ingest_target, settings=settings, recreate=rebuild)
            if result["chunks"]:
                st.success(
                    f"Saved to `{saved_path}` and stored {result['chunks']} chunks "
                    f"from {result['books']} PDF(s). Ask a question now."
                )
            else:
                st.warning(
                    "Saved the file, but no chunks were created. Check that the PDF has text."
                )
        except Exception as exc:  # noqa: BLE001 - show setup issues directly in UI
            st.error(f"Upload/ingest failed: {exc}")

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
                    if s.get("snippet"):
                        st.caption(s["snippet"])

if question := st.chat_input("Ask about RAG, agents, retrieval…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving + reasoning…"):
                result = _answer_once(question)
            st.markdown(result["generation"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        st.markdown(f"**[{s['n']}]** {s['tag']}")
                        if s.get("snippet"):
                            st.caption(s["snippet"])
            st.session_state.messages.append(
                {"role": "assistant", "content": result["generation"], "sources": result["sources"]}
            )
        except Exception as exc:  # noqa: BLE001 — surface provider/runtime errors in-UI
            st.error(
                f"Generation failed: {exc}\n\n"
                "Check that your LLM provider is configured (Ollama running, or an "
                "API key set in `.env`)."
            )
