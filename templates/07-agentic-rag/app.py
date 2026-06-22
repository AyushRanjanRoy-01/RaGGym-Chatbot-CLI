"""
07 · Agentic RAG
ReAct agent that decides when to retrieve from documents vs. when to calculate.
Run `python ingest.py` first to populate the vector store.
"""
import os
import re
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent


# ── Provider helpers ────────────────────────────────────────────────────────

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


# ── Vector store retriever ───────────────────────────────────────────────────

@st.cache_resource
def load_retriever():
    vector_store = os.getenv("VECTOR_STORE", "qdrant")
    embeddings = get_embeddings()
    if vector_store == "chroma":
        from langchain_chroma import Chroma
        db = Chroma(
            persist_directory="./vectorstore_chroma",
            embedding_function=embeddings,
            collection_name="financial_docs",
        )
    else:
        from qdrant_client import QdrantClient
        from langchain_qdrant import QdrantVectorStore
        client = QdrantClient(path="./vectorstore")
        db = QdrantVectorStore(
            client=client,
            collection_name="financial_docs",
            embedding=embeddings,
        )
    return db.as_retriever(search_kwargs={"k": 4})


# ── Tool 2: safe financial calculator ───────────────────────────────────────

@tool
def financial_calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression for financial calculations.
    Supports +, -, *, /, **, (, ), %, and decimal numbers.
    Use for growth rates, margins, ratios — e.g. '(5100000 - 4200000) / 4200000 * 100'.
    """
    # Only allow digits, operators, parentheses, spaces, dots — no letters
    if not re.fullmatch(r"[\d\s\+\-\*\/\(\)\.\%\*]+", expression):
        return "Error: expression contains invalid characters. Use only numbers and operators."
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307 — input sanitised above
        return f"{result:.6g}"
    except Exception as exc:
        return f"Calculation error: {exc}"


# ── Agent factory ────────────────────────────────────────────────────────────

@st.cache_resource
def build_agent():
    retriever = load_retriever()
    search_tool = create_retriever_tool(
        retriever,
        name="search_financial_docs",
        description=(
            "Search the financial documents for facts: revenue figures, net income, EBITDA, "
            "headcount, margins, or any qualitative information about the company. "
            "Use this BEFORE answering any question about reported numbers."
        ),
    )
    system_prompt = (
        "You are a financial analyst assistant. You have two tools:\n"
        "1. search_financial_docs — use this to look up facts from financial reports.\n"
        "2. financial_calculator — use this to compute arithmetic (growth rates, margins, ratios).\n\n"
        "Strategy: search for the raw numbers first, then calculate derived metrics. "
        "Always cite the source figures in your final answer."
    )
    llm = get_llm()
    return create_react_agent(llm, tools=[search_tool, financial_calculator], prompt=system_prompt)


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="07 · Agentic RAG", page_icon="🤖")
st.title("07 · Agentic RAG")
st.caption("ReAct agent: retrieves facts from docs, calculates derived metrics on demand.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the financials (e.g. 'What was QoQ revenue growth in Q2 2024?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent = build_agent()
            result = agent.invoke({"messages": [("human", prompt)]})

            # Collect intermediate tool steps for the expander
            steps = result.get("messages", [])
            tool_calls_log = []
            for step in steps:
                role = getattr(step, "type", "") or step.__class__.__name__
                if role in ("tool", "ToolMessage"):
                    tool_calls_log.append(
                        f"**Tool:** {getattr(step, 'name', '?')}\n```\n{step.content}\n```"
                    )

            if tool_calls_log:
                with st.expander("🔧 Tool calls", expanded=False):
                    st.markdown("\n\n".join(tool_calls_log))

            # Final answer is the last AIMessage
            final = result["messages"][-1].content
            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})

        except FileNotFoundError:
            err = "Vector store not found. Run `python ingest.py` first."
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        except Exception as exc:
            err = f"Error: {exc}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
