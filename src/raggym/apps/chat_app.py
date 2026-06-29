"""Streamlit chat UI for RAGGym — grounded, streamed Q&A over the book corpus.

Launch with ``raggym chat`` (or ``streamlit run src/raggym/apps/chat_app.py``).
"""

from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from raggym.agents import stream_answer
from raggym.config import get_settings
from raggym.ingestion import ingest_path

settings = get_settings()

st.set_page_config(
    page_title="RAGGym · Chat",
    page_icon="🏋️",
    layout="centered",
    initial_sidebar_state="expanded",
)

ASSISTANT_AVATAR = "🏋️"
USER_AVATAR = "🧑‍💻"
EXAMPLES = [
    "What is the ReAct pattern?",
    "Explain prompt chaining with an example",
    "How does reflection improve agent outputs?",
    "Tool use vs. RAG — when to use which?",
]

# ── Styling (light, self-contained — no external assets) ─────────────────────
st.markdown(
    """
    <style>
      #MainMenu, footer, [data-testid="stStatusWidget"] {visibility: hidden;}
      .block-container {padding-top: 2.2rem; padding-bottom: 6rem; max-width: 820px;}
      .rg-hero {text-align: center; margin-bottom: .35rem;}
      .rg-hero h1 {
        font-size: 2.15rem; font-weight: 800; margin: 0; letter-spacing: -.02em;
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      }
      .rg-hero p {color: #8a8aa0; margin: .3rem 0 0; font-size: .98rem;}
      [data-testid="stChatMessage"] {
        border-radius: 16px; padding: .35rem .25rem; margin-bottom: .15rem;
      }
      [data-testid="stChatMessage"] p {line-height: 1.6;}
      .rg-src {
        font-size: .82rem; color: #6b6b80; border-left: 2px solid #8b5cf6;
        padding-left: .6rem; margin: .35rem 0;
      }
      .stButton button {border-radius: 10px; font-weight: 500;}
      div[data-testid="stChatInput"] textarea {border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helpers ──────────────────────────────────────────────────────────────────
def _safe_pdf_name(filename: str) -> str:
    stem = Path(filename).name.strip() or "upload.pdf"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", stem)


def _save_upload(uploaded_file) -> Path:
    settings.books_dir.mkdir(parents=True, exist_ok=True)
    target = settings.books_dir / _safe_pdf_name(uploaded_file.name)
    target.write_bytes(uploaded_file.getbuffer())
    return target


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['n']}] {s['tag']}**")
            if s.get("snippet"):
                st.markdown(f"<div class='rg-src'>{s['snippet']}</div>", unsafe_allow_html=True)


def _ask(prompt: str) -> None:
    """Render the user turn, then stream the assistant answer + sources."""
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            events = stream_answer(prompt, settings=settings)
            captured = {"sources": []}
            first_token = None
            with st.spinner("Searching the corpus…"):
                for kind, payload in events:
                    if kind == "sources":
                        captured["sources"] = payload
                    elif kind == "token":
                        first_token = payload
                        break

            def _tokens():
                if first_token:
                    yield first_token
                for kind, payload in events:
                    if kind == "token":
                        yield payload

            answer_text = st.write_stream(_tokens())
            _render_sources(captured["sources"])
            st.session_state.messages.append(
                {"role": "assistant", "content": answer_text, "sources": captured["sources"]}
            )
        except Exception as exc:  # noqa: BLE001 — surface provider/setup errors in-UI
            st.error(
                f"Generation failed: {exc}\n\n"
                "Check that a corpus is ingested and an LLM provider is configured "
                "(run Ollama, or set an API key in `.env`)."
            )


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏋️ RAGGym")
    st.caption("Grounded Q&A over your RAG book corpus.")

    with st.expander("⚙️ Configuration", expanded=False):
        st.write(
            {
                "llm": f"{settings.llm_provider}:{settings.llm_model}",
                "embeddings": f"{settings.embed_provider}:{settings.embed_model}",
                "vector_store": settings.vector_store,
                "hybrid": settings.use_hybrid,
                "reranker": settings.use_reranker,
                "corrective": settings.use_corrective,
                "top_k": settings.retrieval_top_k,
            }
        )
        if settings.llm_provider == "ollama":
            st.info("Local Ollama — ensure `ollama serve` is running and the model is pulled.")

    with st.expander("📤 Add a book (PDF)", expanded=False):
        uploaded_pdf = st.file_uploader(
            f"Saved to `{settings.books_dir}`", type=["pdf"], accept_multiple_files=False
        )
        rebuild = st.checkbox(
            "Rebuild vector DB from all saved PDFs",
            value=True,
            help="Recommended after changing embedding provider/model; avoids duplicate chunks.",
        )
        if uploaded_pdf and st.button("Save + ingest", type="primary", use_container_width=True):
            try:
                saved_path = _save_upload(uploaded_pdf)
                with st.spinner("Chunking, embedding, and storing…"):
                    ingest_target = settings.books_dir if rebuild else saved_path
                    result = ingest_path(ingest_target, settings=settings, recreate=rebuild)
                if result["chunks"]:
                    captions = sum(file.get("visual_captions", 0) for file in result["files"])
                    st.success(
                        f"Stored {result['chunks']} chunks from {result['books']} PDF(s), "
                        f"including {captions} visual caption(s). Ask away!"
                    )
                else:
                    st.warning(
                        "Saved the file, but no chunks were created — does the PDF have text?"
                    )
            except Exception as exc:  # noqa: BLE001 — show setup issues directly in UI
                st.error(f"Upload/ingest failed: {exc}")

    if st.session_state.get("messages"):
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='rg-hero'><h1>🏋️ RAGGym</h1>"
    "<p>Ask anything about RAG, agents, and retrieval — "
    "answers are grounded in your corpus.</p></div>",
    unsafe_allow_html=True,
)

# ── State ────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Empty state: clickable example prompts ───────────────────────────────────
if not st.session_state.messages:
    st.caption("Try one of these:")
    cols = st.columns(2)
    for i, example in enumerate(EXAMPLES):
        if cols[i % 2].button(example, key=f"ex_{i}", use_container_width=True):
            _ask(example)
            st.rerun()

# ── History ──────────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = ASSISTANT_AVATAR if msg["role"] == "assistant" else USER_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            _render_sources(msg.get("sources", []))

# ── Input ────────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about RAG, agents, retrieval…"):
    _ask(prompt)
    st.rerun()
