import streamlit as st

from core.logging import setup_logging
from config import settings
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore
from rag.chain import build_rag_chain

setup_logging()


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def _load_pipeline():
    embeddings = get_embeddings()
    store = get_vectorstore(embeddings)
    retriever = store.as_retriever()
    return build_rag_chain(retriever)


def main():
    st.set_page_config(
        page_title=settings.app_title,
        page_icon="💼",
        layout="centered",
    )
    st.title(settings.app_title)
    st.caption(settings.app_description)

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
