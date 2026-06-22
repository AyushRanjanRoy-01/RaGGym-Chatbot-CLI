import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def get_llm():
    if os.getenv("LLM_PROVIDER", "openai") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("LLM_MODEL", "llama3.2:3b"),
                          base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)


def get_embeddings():
    if os.getenv("EMBED_PROVIDER", "openai") == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
                                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))


@st.cache_resource
def build_retriever():
    vector_store_type = os.getenv("VECTOR_STORE", "qdrant")
    embeddings = get_embeddings()
    if vector_store_type == "chroma":
        from langchain_chroma import Chroma
        store = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    else:
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        client = QdrantClient(path="./vectorstore")
        store = QdrantVectorStore(client=client, collection_name="financial_docs",
                                  embedding=embeddings)
    return store.as_retriever(search_kwargs={"k": 4})


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a financial analyst assistant. Answer ONLY from the provided context. "
     "Cite the source filename in square brackets after each fact, e.g. [sample_financials.txt]. "
     "If the answer is not in the context, say 'I don't have that information in the provided documents.'"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

# LCEL RAG chain: parallel fetch context + pass question, then prompt → LLM → parse
chain = (
    {"context": build_retriever() | format_docs, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | get_llm()
    | StrOutputParser()
)

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("Document Q&A — Financial RAG")
st.caption("Run `python ingest.py` first to index your docs.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about your financial documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        answer = st.write_stream(chain.stream(question))

    st.session_state.messages.append({"role": "assistant", "content": answer})
