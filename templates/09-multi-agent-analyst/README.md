# 09 · Multi-Agent Financial Analyst

Three specialized LangGraph agents produce an inspectable financial brief: **Researcher → Analyst → Writer**.

---

## What this demonstrates

- **Separation of concerns** — retrieval, reasoning, and writing are independent nodes with separate prompts, making each step tunable without touching the others.
- **Inspectable intermediate outputs** — every node's output is visible in the UI, so you can audit exactly what context was retrieved and how it was interpreted before the final brief is generated.
- **Why it beats one mega-prompt** — a single prompt that retrieves, analyses, and writes simultaneously produces opaque outputs with no clear failure point. Three nodes mean you can swap the analyst prompt without re-tuning retrieval, replace the writer LLM with a fine-tuned model, or add a compliance-check node between analyst and writer.
- **Finance relevance** — earnings summaries, risk extraction, and structured briefs are high-frequency tasks in investment research and corporate finance; this pattern maps directly to those workflows.

---

## Architecture

```
START
  │
  ▼
[researcher]  ── vector search → retrieves top-k chunks → state["research"]
  │
  ▼
[analyst]     ── LLM extracts figures / trends / risks → state["analysis"]
  │
  ▼
[writer]      ── LLM writes markdown brief (Summary / Key Figures / Risks / Outlook)
  │                                                      → state["report"]
  ▼
END
```

**State** (`TypedDict`): `question`, `research`, `analysis`, `report`

---

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# --- Ollama alternative (no API key needed) ---
# ollama pull llama3.2:3b
# ollama pull nomic-embed-text
# Set LLM_PROVIDER=ollama and EMBED_PROVIDER=ollama in .env
```

---

## Run

```powershell
# Step 1 — ingest sample documents
python ingest.py

# Step 2 — launch the Streamlit app
streamlit run app.py
```

---

## How it works

1. **Researcher node** — calls the vector store retriever with the user's question, joins the top-5 chunks into a single `research` string, and stores it in state. This is the only node that touches the database.

2. **Analyst node** — sends a structured prompt to the LLM asking it to extract key financial figures, YoY/QoQ trends, and risk watchpoints from the raw `research` text. Output stored as `analysis`. No retrieval happens here — separation is strict.

3. **Writer node** — takes the `analysis` and converts it into a polished markdown brief with fixed sections (`## Summary`, `## Key Figures`, `## Risks`, `## Outlook`). This node only knows how to write; it does not re-interpret raw data.

4. **Streamlit UI** — runs the full graph and surfaces each node's output in an `st.expander` so you can inspect the decision chain before reading the final report.

---

## When to use this pattern

- You need **auditable intermediate steps** (compliance, model governance).
- Different nodes benefit from **different models** (cheap LLM for writing, powerful LLM for analysis).
- You want to **add nodes** (e.g. a fact-checker or a chart-generator) without rewriting the pipeline.
- The task is complex enough that a single prompt produces inconsistent structure.

---

## Tech stack

LangGraph 0.2 · LangChain 0.3 · Qdrant (local) · OpenAI / Ollama · Streamlit
