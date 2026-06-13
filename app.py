import re
from pathlib import Path

import streamlit as st

from core.logging import setup_logging
from config import settings
from rag.chunkers import RecursiveChunker
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore
from rag.chain import build_rag_chain
from rag.loaders import TextLoader

setup_logging()

APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR / "docs"


@st.cache_resource(show_spinner="Loading embeddings...")
def _get_embeddings():
    return get_embeddings()


@st.cache_resource(show_spinner="Connecting vector store...")
def _get_store():
    return get_vectorstore(_get_embeddings())


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def _load_pipeline():
    retriever = _get_store().as_retriever()
    return build_rag_chain(retriever)


def _sanitize_filename(filename: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
    return safe_name or "uploaded.txt"


def _build_target_path(filename: str) -> Path:
    target_path = DOCS_DIR / _sanitize_filename(filename)
    stem = target_path.stem
    suffix = target_path.suffix or ".txt"
    counter = 1

    while target_path.exists():
        target_path = DOCS_DIR / f"{stem}_{counter}{suffix}"
        counter += 1

    return target_path


def _save_document(uploaded_file) -> Path:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = _build_target_path(uploaded_file.name)
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def _save_and_index_document(uploaded_file) -> tuple[Path, int]:
    target_path = _save_document(uploaded_file)
    documents = TextLoader().load(str(target_path))
    chunks = RecursiveChunker().chunk(documents)
    _get_store().add_documents(chunks)
    return target_path, len(chunks)


def main():
    st.set_page_config(
        page_title=settings.app_title,
        page_icon="💼",
        layout="centered",
    )
    st.title(settings.app_title)
    st.caption(settings.app_description)

    if upload_notice := st.session_state.pop("upload_notice", None):
        st.success(upload_notice)

    _, upload_col = st.columns([0.9, 0.1])
    with upload_col:
        with st.popover("📎"):
            with st.form("upload_document_form", clear_on_submit=True):
                uploaded_file = st.file_uploader(
                    "Upload a .txt document",
                    type=["txt"],
                    label_visibility="collapsed",
                )
                submitted = st.form_submit_button("Upload")

            if submitted:
                if uploaded_file is None:
                    st.warning("Choose a .txt file first.")
                else:
                    try:
                        with st.spinner("Saving and indexing document..."):
                            saved_path, chunk_count = _save_and_index_document(uploaded_file)
                        _load_pipeline.clear()
                        st.session_state.upload_notice = (
                            f"Saved {saved_path.relative_to(APP_DIR)} and indexed {chunk_count} chunks. Ready to chat."
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Upload failed: {exc}")

    chain = _load_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        history = st.session_state.messages[:-1]
        with st.chat_message("assistant"):
            response = st.write_stream(
                chain.stream({"question": prompt, "history": history})
            )

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
