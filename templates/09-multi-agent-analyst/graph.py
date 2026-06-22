"""
LangGraph definition for the three-agent analyst pipeline.

Nodes
-----
researcher  → retrieves relevant chunks from the vector store
analyst     → extracts key figures, trends, and risks from raw research
writer      → turns the analysis into a structured markdown brief

Flow: START → researcher → analyst → writer → END
"""
import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()


# ── State schema ──────────────────────────────────────────────────────────────
class AnalystState(TypedDict):
    question: str
    research: str    # raw retrieved context
    analysis: str    # key figures / trends / risks
    report: str      # final markdown brief


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
    return vs.as_retriever(search_kwargs={"k": 5})


# ── Nodes ─────────────────────────────────────────────────────────────────────
def researcher_node(state: AnalystState) -> dict:
    """Retrieve relevant chunks; join them into a single context string."""
    retriever = get_retriever()
    docs = retriever.invoke(state["question"])
    research = "\n\n".join(d.page_content for d in docs)
    return {"research": research}


def analyst_node(state: AnalystState) -> dict:
    """Extract key figures, trends, and risks from the raw research."""
    llm = get_llm()
    prompt = (
        "You are a senior financial analyst. Given the raw research below, extract:\n"
        "- Key financial figures (revenue, margins, EPS, ratios, etc.)\n"
        "- Trends (YoY/QoQ movements, guidance changes)\n"
        "- Risks or watchpoints mentioned\n\n"
        f"Research:\n{state['research']}\n\n"
        "Provide a structured bullet-point analysis."
    )
    analysis = llm.invoke(prompt).content
    return {"analysis": analysis}


def writer_node(state: AnalystState) -> dict:
    """Turn the structured analysis into a polished markdown brief."""
    llm = get_llm()
    prompt = (
        "You are a financial writer. Convert the analysis below into a concise markdown brief "
        "with these exact sections: ## Summary, ## Key Figures, ## Risks, ## Outlook.\n\n"
        f"Question: {state['question']}\n\n"
        f"Analysis:\n{state['analysis']}\n\n"
        "Return only the markdown brief."
    )
    report = llm.invoke(prompt).content
    return {"report": report}


# ── Graph assembly ────────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(AnalystState)
    g.add_node("researcher", researcher_node)
    g.add_node("analyst", analyst_node)
    g.add_node("writer", writer_node)
    g.add_edge(START, "researcher")
    g.add_edge("researcher", "analyst")
    g.add_edge("analyst", "writer")
    g.add_edge("writer", END)
    return g.compile()
