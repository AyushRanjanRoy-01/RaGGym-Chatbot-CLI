# 06 · Multi-Query RAG (RAG-Fusion)

LLM expands one question into 4 variants, retrieves each, fuses results with Reciprocal Rank Fusion (RRF), then answers from the fused top documents.

## What this demonstrates

- **Single-phrasing fragility**: "What is the company's profitability?" may miss the chunk mentioning "Net Income" or "EBITDA margin" because the embedding distances differ — a single query under-retrieves in financial corpora with varied terminology.
- **Query expansion via LLM**: generating multiple phrasings of the same question casts a wider net over the embedding space.
- **Manual RRF implementation**: the teaching point — documents that rank highly across multiple query variants get promoted; noise that ranks well for only one variant gets demoted.
- **Transparent expansion**: the Streamlit UI shows the 4 generated variants so users see what was searched.

## Architecture

```
User question
    │
    ▼
LLM (query expansion) → 4 query variants
    │
    ├─ variant 1 → Retriever → [doc list 1]
    ├─ variant 2 → Retriever → [doc list 2]
    ├─ variant 3 → Retriever → [doc list 3]
    └─ variant 4 → Retriever → [doc list 4]
                                     │
                                     ▼
                          Reciprocal Rank Fusion (k=60)
                                     │
                                     ▼
                          Top-5 fused docs
                                     │
                                     ▼
                          ChatPromptTemplate | LLM | StrOutputParser
                                     │
                                     ▼
                          Streamlit chat UI (streams answer + shows variants)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env         # then add your OPENAI_API_KEY
```

**Ollama backup** (no API key needed):
```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# set LLM_PROVIDER=ollama and EMBED_PROVIDER=ollama in .env
```

**Chroma fallback**: set `VECTOR_STORE=chroma` in `.env`.

## Run

```powershell
python ingest.py
streamlit run app.py
```

## How it works

1. `ingest.py` loads `.txt` files from `docs/`, chunks and embeds them, stores in Qdrant (or Chroma).
2. `app.py` sends the user question to the LLM with a prompt asking for 4 distinct search query variants.
3. Each variant is sent independently to the same retriever (k=5 per variant), yielding up to 4 × 5 = 20 candidate chunks (with duplicates).
4. **Reciprocal Rank Fusion** (`1 / (k + rank)` with k=60) merges the 4 ranked lists: a chunk appearing at rank 1 in all 4 lists scores much higher than one appearing at rank 1 in only 1 list. This is the core teaching point.
5. The top-5 fused (deduplicated) chunks are passed to an answer prompt.
6. The final answer is streamed via `st.write_stream`; generated query variants are shown in an expander.

## When to use this pattern

Use RAG-Fusion when questions are **ambiguous or broad** (e.g. "how is the company doing?"), when documents use **mixed terminology** (net income vs. profit vs. earnings), or when a single phrasing consistently under-retrieves. The extra LLM call for expansion adds ~1–2s latency but meaningfully improves recall on financial Q&A.

## Tech stack

Qdrant / Chroma · LangChain LCEL · Reciprocal Rank Fusion (manual) · OpenAI/Ollama · Streamlit
