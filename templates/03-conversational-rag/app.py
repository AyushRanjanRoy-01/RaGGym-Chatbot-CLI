import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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
def build_rag_chain():
    # ── 1. Vector store retriever ─────────────────────────────────────────────
    embeddings = get_embeddings()
    if os.getenv("VECTOR_STORE", "qdrant") == "chroma":
        from langchain_chroma import Chroma
        store = Chroma(persist_directory="./vectorstore", embedding_function=embeddings)
    else:
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        client = QdrantClient(path="./vectorstore")
        store = QdrantVectorStore(client=client, collection_name="financial_docs",
                                  embedding=embeddings)
    retriever = store.as_retriever(search_kwargs={"k": 4})

    llm = get_llm()

    # ── 2. Contextualize sub-chain: rewrite follow-up into a standalone question
    # Without this step, "what about the prior quarter?" would be sent to the
    # retriever verbatim and return irrelevant chunks.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, reformulate the question "
         "into a standalone question that can be understood without the chat history. "
         "Do NOT answer the question — only rewrite it if needed, otherwise return it as-is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    # ── 3. Answer chain: answer using retrieved context + full history ─────────
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a financial analyst assistant. Answer ONLY from the context below. "
         "Cite the source filename in square brackets after each fact. "
         "If the answer is not in the context, say 'I don't have that information.'\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)

    # ── 4. Full RAG chain: history-aware retrieval → stuff documents → answer ──
    return create_retrieval_chain(history_aware_retriever, answer_chain)


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("Conversational Financial RAG")
st.caption("History-aware retrieval — follow-up questions work correctly.")

rag_chain = build_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []       # {"role": str, "content": str}
if "lc_history" not in st.session_state:
    st.session_state.lc_history = []     # LangChain HumanMessage / AIMessage objects

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if question := st.chat_input("Ask about your financial documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        # Stream the answer key from the retrieval chain's output dict
        response_chunks = []
        placeholder = st.empty()
        for chunk in rag_chain.stream(
            {"input": question, "chat_history": st.session_state.lc_history}
        ):
            if "answer" in chunk:          # only the answer key carries text tokens
                response_chunks.append(chunk["answer"])
                placeholder.write("".join(response_chunks))
        answer = "".join(response_chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.lc_history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])
