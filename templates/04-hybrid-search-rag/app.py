import os
import streamlit as st
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def get_llm():
    if os.getenv("LLM_PROVIDER", "openai") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3.2:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)


def get_embeddings():
    if os.getenv("EMBED_PROVIDER", "openai") == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))


@st.cache_resource
def load_retriever():
    embeddings = get_embeddings()
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")
    vs = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse,
        path="./vectorstore",
        collection_name="financials_hybrid",
        vector_name="dense",
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,  # fuses dense cosine + BM25 scores via Qdrant RRF
    )
    return vs.as_retriever(search_kwargs={"k": 5})


prompt = ChatPromptTemplate.from_template(
    "You are a financial analyst assistant.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
)


def build_chain(retriever, llm):
    from langchain_core.runnables import RunnablePassthrough

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


st.title("04 · Hybrid Search RAG")
st.caption("Dense + sparse (BM25) hybrid retrieval — exact financial terms meet semantic search.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about the financials (try: 'What is the Basel III CET1 ratio?')"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    retriever = load_retriever()
    chain = build_chain(retriever, get_llm())

    with st.chat_message("assistant"):
        response = st.write_stream(chain.stream(question))

    st.session_state.messages.append({"role": "assistant", "content": response})
