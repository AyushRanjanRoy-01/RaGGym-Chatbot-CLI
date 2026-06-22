# 10 · Corrective RAG (CRAG)

Self-correcting RAG that grades retrieved documents for relevance, rewrites the query if they are poor, and retries before generating — capping loops at 2 retries.

---

## What this demonstrates

- **The naive-RAG problem** — standard RAG answers from whatever it retrieves, even if the documents are irrelevant. In financial analysis, answering from the wrong context produces hallucinated figures (fabricated revenue numbers, wrong ratios) that can be indistinguishable from real ones.
- **CRAG's correction loop** — an LLM grader decides whether each retrieved document is actually relevant. If too few are, the query is rewritten and retrieval retries. Only then is the answer generated.
- **Bounded retries** — the loop is capped (`MAX_RETRIES=2`) to avoid infinite cycles. If retries are exhausted, the answer is generated from whatever was found (or from an empty context, which the model handles gracefully).
- **Transparency** — every grading decision and every query rewrite is visible in the UI, making the self-correction auditable.

---

## Architecture

```
START
  │
  ▼
[retrieve]          ── vector search with current question
  │
  ▼
[grade_documents]   ── LLM grades each doc YES/NO; keeps relevant ones
  │
  ├── enough relevant docs  ──────────────────────────────┐
  │   OR retries exhausted                                │
  │                                                       ▼
  └── too few docs AND retries < 2    [generate]  ── answers from kept docs → END
          │
          ▼
   [transform_query]   ── LLM rewrites the question
          │
          │  [WEB-SEARCH EXTENSION POINT]
          │  In production: add a web_search_node here
          │  to fetch live data when local docs are insufficient.
          │
          └──→  back to [retrieve]  (loop)
```

**State** (`TypedDict`): `question`, `original_question`, `documents`, `generation`, `retries`

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

1. **retrieve node** — performs vector similarity search using the current question (which may have been rewritten by a prior loop iteration).

2. **grade_documents node** — for each retrieved document, an LLM answers "YES or NO: is this relevant to the question?" Only YES documents are kept. This is the key CRAG innovation: don't trust retrieval blindly.

3. **Router** (`add_conditional_edges`) — checks two conditions:
   - If `len(relevant_docs) >= 1` **or** `retries >= 2`: proceed to **generate**.
   - Otherwise: proceed to **transform_query**.

4. **transform_query node** — rewrites the question to be more specific or use different vocabulary, then increments the retry counter. *This is also where a web-search node would plug in* — clearly labeled in the code — to fetch live financial data when the local vector store comes up empty.

5. **generate node** — constructs an answer from the kept documents, clearly noting if no relevant context was found. Always answers the `original_question`, not the rewritten one.

6. **Streamlit UI** — streams each graph step and shows: retrieved docs, grading verdicts, any rewritten queries (with the web-search extension point called out), and the final answer.

---

## When to use this pattern

- Financial Q&A where hallucinated figures (wrong EPS, wrong ratios) are unacceptable.
- Any RAG system where documents have variable relevance and you want the model to self-correct rather than answer from noise.
- As a foundation: add the web-search node at the labeled extension point to handle queries that the local corpus cannot answer.

---

## Tech stack

LangGraph 0.2 · LangChain 0.3 · Qdrant (local) · OpenAI / Ollama · Streamlit
