# 07 · Agentic RAG

A ReAct agent that **decides when to retrieve** from financial documents vs. **when to compute** — rather than blindly retrieving on every query.

## What this demonstrates

- **Agent-driven retrieval**: The agent only calls the retriever when it judges that facts are missing — saving tokens and latency on pure-math questions.
- **Tool chaining**: For questions like *"What was QoQ revenue growth in Q2?"*, the agent chains retrieve → calculate automatically.
- **Safe eval for finance**: A restricted arithmetic evaluator handles growth rates, margins, and ratios without executing arbitrary code.
- **Agentic RAG vs static RAG**: Static RAG always retrieves; agentic RAG retrieves only when needed and can combine retrieval with computation in a single reasoning loop.

## Architecture

```
User question
      │
      ▼
 ReAct Agent  (LLM + langgraph.prebuilt.create_react_agent)
      │
      ├──► search_financial_docs ──► Qdrant / Chroma vector store
      │         (facts, figures)           └──► ./docs/*.txt
      │
      └──► financial_calculator
                (arithmetic only)
```

The agent reasons step-by-step, deciding at each step which tool to call (or none).

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and set OPENAI_API_KEY

# 4. Ingest documents into the vector store
python ingest.py
```

**Ollama backup** (no OpenAI key required):
```powershell
# Pull models first
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# In .env set LLM_PROVIDER=ollama, EMBED_PROVIDER=ollama
```

## Run

```powershell
streamlit run app.py
```

## How it works

1. User sends a question via Streamlit chat.
2. `create_react_agent` runs the ReAct loop: the LLM produces a thought, chooses a tool (or none), receives the result, and repeats.
3. **`search_financial_docs`** (Tool 1) — a `create_retriever_tool` wrapper around a Qdrant/Chroma retriever; returns the top-4 relevant chunks from ingested `.txt` files.
4. **`financial_calculator`** (Tool 2) — validates that the expression contains only digits and arithmetic operators, then calls `eval` in a sandboxed namespace.
5. Tool calls and results are shown in a collapsible expander; the final answer is rendered in the chat.
6. The vector store is created by `ingest.py`, which chunks documents with `RecursiveCharacterTextSplitter(1000/200)` and derives embedding dimension dynamically.

## When to use this pattern

- You have **both structured facts (in docs) and derived metrics** to answer (growth rates, margins, ratios).
- You want **selective retrieval** — the agent skips the retriever for simple math-only questions, cutting latency.
- You need a **transparent reasoning trace** (tool calls visible in the UI) for compliance or debugging.
- Natural extension: add more tools (SQL query, web search, Python exec) while keeping the same ReAct backbone.

## Tech stack

LangChain 0.3 · LangGraph 0.2 (`create_react_agent`) · Qdrant local · OpenAI / Ollama · Streamlit
