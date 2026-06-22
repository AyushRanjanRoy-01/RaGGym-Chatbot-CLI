import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

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


def get_base_vector_store(embeddings):
    if os.getenv("VECTOR_STORE", "qdrant") == "chroma":
        from langchain_chroma import Chroma
        return Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    from langchain_qdrant import QdrantVectorStore
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings, path="./vectorstore", collection_name="financials_rerank"
    )


@st.cache_resource
def load_retriever():
    embeddings = get_embeddings()
    vs = get_base_vector_store(embeddings)
    base_retriever = vs.as_retriever(search_kwargs={"k": 20})  # cast wide with bi-encoder

    # Cross-encoder scores every (query, chunk) pair jointly — much more precise than bi-encoder
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=4)
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


prompt = ChatPromptTemplate.from_template(
    "You are a financial analyst assistant.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
)


def build_chain(retriever, llm):
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


st.title("05 · Reranking RAG")
st.caption("Retrieve wide (k=20) with a bi-encoder, rerank to top 4 with a cross-encoder.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about the financials (try: 'What is the EPS and how does it compare to consensus?')"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    retriever = load_retriever()
    chain = build_chain(retriever, get_llm())

    with st.chat_message("assistant"):
        response = st.write_stream(chain.stream(question))

    st.session_state.messages.append({"role": "assistant", "content": response})
