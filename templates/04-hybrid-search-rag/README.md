# 04 · Hybrid Search RAG

Dense + sparse (BM25) hybrid retrieval for financial document Q&A.

## What this demonstrates

- **Exact-term recall**: financial jargon like `EBITDA`, ticker symbols (`ACF`), and regulatory labels (`Basel III`) are rare tokens that pure semantic search often misses — BM25 catches them exactly.
- **Semantic coverage**: dense embeddings handle paraphrase, context, and meaning beyond surface tokens.
- **Qdrant native fusion**: Qdrant fuses dense cosine similarity and BM25 sparse scores internally using Reciprocal Rank Fusion (RRF), returning a single ranked list.
- **Why Qdrant only**: Chroma does not support sparse/hybrid vector search; this template is Qdrant-only by design.

## Architecture

```
docs/*.txt
    │
    ▼
DirectoryLoader → RecursiveCharacterTextSplitter
    │
    ▼
QdrantVectorStore (HYBRID mode)
    ├─ dense vector  (OpenAI / Ollama embeddings)
    └─ sparse vector (FastEmbedSparse / BM25)
    │
    ▼  Qdrant RRF fusion
Retriever (k=5)
    │
    ▼
ChatPromptTemplate | LLM | StrOutputParser
    │
    ▼
Streamlit chat UI
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

## Run

```powershell
python ingest.py
streamlit run app.py
```

## How it works

1. `ingest.py` loads `.txt` files from `docs/`, splits them into ~400-token chunks.
2. Each chunk is encoded with **both** a dense embedding (OpenAI/Ollama) and a BM25 sparse vector (FastEmbedSparse).
3. Both vector types are stored in a single Qdrant collection created with `RetrievalMode.HYBRID`.
4. At query time, Qdrant runs dense ANN search and BM25 sparse search in parallel, then merges results with **Reciprocal Rank Fusion** — chunks that rank well in both lists float to the top.
5. The top-5 fused chunks are passed to the LLM via a `ChatPromptTemplate`.
6. The Streamlit UI streams the answer token-by-token with `st.write_stream`.

## When to use this pattern

Use hybrid search when your documents contain **exact financial identifiers** (tickers, CUSIP, ISIN, regulatory ratios, named standards like Basel III/IFRS 9) alongside prose. Pure dense retrieval loses precision on rare tokens; pure BM25 loses on paraphrase. Hybrid covers both.

## Tech stack

Qdrant (hybrid RRF) · FastEmbedSparse (BM25) · LangChain LCEL · OpenAI/Ollama · Streamlit
