"""
Corrective RAG (CRAG) LangGraph definition.

Nodes
-----
retrieve          → vector search for the current question
grade_documents   → LLM grades each doc; keeps relevant ones; sets a flag
transform_query   → LLM rewrites the question when docs are insufficient
generate          → LLM answers from the (possibly corrected) documents

Routing
-------
After grade_documents:
  - enough relevant docs OR retries exhausted  →  generate
  - too few relevant docs AND retries < MAX    →  transform_query → retrieve (loop)

MAX_RETRIES = 2  (caps the retrieve/rewrite loop)

Web-search extension point
--------------------------
In a production CRAG you could replace or supplement transform_query→retrieve
with a web-search node.  The slot is clearly marked below.
"""
import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

load_dotenv()

MAX_RETRIES = 2
RELEVANCE_THRESHOLD = 1   # minimum relevant docs required to skip rewrite


# ── State schema ──────────────────────────────────────────────────────────────
class CRAGState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    generation: str
    retries: int


# ── Provider helpers ──────────────────────────────────────────────────────────
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


def get_retriever():
    embeddings = get_embeddings()
    if os.getenv("VECTOR_STORE", "qdrant") == "chroma":
        from langchain_chroma import Chroma
        vs = Chroma(persist_directory="./vectorstore", embedding_function=embeddings,
                    collection_name="financials")
    else:
        from langchain_qdrant import QdrantVectorStore
        vs = QdrantVectorStore.from_existing_collection(
            embedding=embeddings, path="./vectorstore", collection_name="financials"
        )
    return vs.as_retriever(search_kwargs={"k": 4})


# ── Nodes ─────────────────────────────────────────────────────────────────────
def retrieve_node(state: CRAGState) -> dict:
    """Vector search using the current (possibly rewritten) question."""
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    return {"documents": docs}


def grade_documents_node(state: CRAGState) -> dict:
    """LLM binary-grades each document; keeps only relevant ones."""
    llm = get_llm()
    question = state["question"]
    relevant = []
    for doc in state["documents"]:
        verdict = llm.invoke(
            f"Is the following document relevant to the question?\n"
            f"Question: {question}\n"
            f"Document: {doc.page_content[:600]}\n"
            "Reply with a single word: YES or NO."
        ).content.strip().upper()
        if verdict.startswith("YES"):
            relevant.append(doc)
    return {"documents": relevant}


def transform_query_node(state: CRAGState) -> dict:
    """Rewrite the question to improve retrieval recall."""
    llm = get_llm()
    # [WEB-SEARCH EXTENSION POINT]
    # In a production system you could add a web_search_node here that runs
    # alongside (or instead of) the rewritten local query, e.g.:
    #   web_docs = web_search(state["question"])
    #   return {"documents": web_docs, "retries": state["retries"] + 1}
    new_question = llm.invoke(
        f"The original question did not return useful documents.\n"
        f"Rewrite it to be more specific so a vector search will find better matches.\n"
        f"Original question: {state['original_question']}\n"
        f"Current question: {state['question']}\n"
        "Return only the rewritten question."
    ).content.strip()
    return {"question": new_question, "retries": state["retries"] + 1}


def generate_node(state: CRAGState) -> dict:
    """Answer the original question using whatever documents were kept."""
    llm = get_llm()
    context = "\n\n".join(d.page_content for d in state["documents"])
    if not context:
        context = "No relevant documents were found."
    answer = llm.invoke(
        f"You are a financial analyst assistant. Answer the question using only the context below.\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['original_question']}\n"
        "Answer concisely and cite specific figures where available."
    ).content
    return {"generation": answer}


# ── Router ────────────────────────────────────────────────────────────────────
def route_after_grading(state: CRAGState) -> str:
    """Decide whether to generate or rewrite-and-retry."""
    enough_docs = len(state["documents"]) >= RELEVANCE_THRESHOLD
    retries_exhausted = state["retries"] >= MAX_RETRIES
    if enough_docs or retries_exhausted:
        return "generate"
    return "transform_query"


# ── Graph assembly ────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(CRAGState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade_documents", grade_documents_node)
    g.add_node("transform_query", transform_query_node)
    g.add_node("generate", generate_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(   # core CRAG routing decision
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "transform_query": "transform_query"},
    )
    g.add_edge("transform_query", "retrieve")   # loop back
    g.add_edge("generate", END)
    return g.compile()
