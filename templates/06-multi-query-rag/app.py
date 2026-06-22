import os
from collections import defaultdict
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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


def get_vector_store(embeddings):
    if os.getenv("VECTOR_STORE", "qdrant") == "chroma":
        from langchain_chroma import Chroma
        return Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    from langchain_qdrant import QdrantVectorStore
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings, path="./vectorstore", collection_name="financials_mq"
    )


def reciprocal_rank_fusion(results_per_query: list[list], k: int = 60) -> list:
    """Merge multiple ranked doc lists into one via RRF. k=60 is the standard constant."""
    scores: dict = defaultdict(float)
    doc_map: dict = {}
    for ranked_list in results_per_query:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content  # deduplicate by content
            scores[key] += 1.0 / (k + rank + 1)  # RRF formula: 1 / (k + rank)
            doc_map[key] = doc
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


# Prompt that asks the LLM to generate 4 search query variants
QUERY_EXPANSION_PROMPT = ChatPromptTemplate.from_template(
    "You are a financial research assistant. Generate exactly 4 distinct search queries "
    "to retrieve information relevant to answering the following question from financial documents.\n"
    "Output ONLY the 4 queries, one per line, no numbering or bullets.\n\n"
    "Question: {question}"
)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are a financial analyst assistant.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer concisely:"
)


@st.cache_resource
def load_retriever():
    embeddings = get_embeddings()
    vs = get_vector_store(embeddings)
    return vs.as_retriever(search_kwargs={"k": 5})


def rag_fusion_chain(question: str, retriever, llm):
    # Step 1: expand question into 4 variants
    expand_chain = QUERY_EXPANSION_PROMPT | llm | StrOutputParser()
    variants_raw = expand_chain.invoke({"question": question})
    variants = [q.strip() for q in variants_raw.strip().splitlines() if q.strip()][:4]

    # Step 2: retrieve docs for each variant independently
    all_results = [retriever.invoke(v) for v in variants]

    # Step 3: fuse with Reciprocal Rank Fusion — this is the teaching point of RAG-Fusion
    fused_docs = reciprocal_rank_fusion(all_results)[:5]

    return variants, fused_docs


def stream_answer(question: str, fused_docs, llm):
    context = "\n\n".join(d.page_content for d in fused_docs)
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
    return answer_chain.stream({"context": context, "question": question})


st.title("06 · Multi-Query RAG (RAG-Fusion)")
st.caption("LLM expands one question into 4 variants → retrieve each → fuse with RRF → answer.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about the financials (try: 'How is the company performing financially?')"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    retriever = load_retriever()
    llm = get_llm()

    with st.spinner("Generating query variants and retrieving..."):
        variants, fused_docs = rag_fusion_chain(question, retriever, llm)

    with st.expander("Query variants generated"):
        for v in variants:
            st.write(f"- {v}")

    with st.chat_message("assistant"):
        response = st.write_stream(stream_answer(question, fused_docs, llm))

    st.session_state.messages.append({"role": "assistant", "content": response})
