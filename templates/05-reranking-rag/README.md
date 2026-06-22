# 05 · Reranking RAG

Retrieve wide with a fast bi-encoder, rerank to a tight top-4 with a cross-encoder.

## What this demonstrates

- **Bi-encoder vs cross-encoder tradeoff**: bi-encoders embed query and doc independently (fast, scalable) but miss fine-grained relevance; cross-encoders score the pair jointly (slower, but much more precise).
- **Recall then precision**: cast a wide net (k=20) for recall, compress to 4 highly relevant chunks for precision before the LLM.
- **`ContextualCompressionRetriever`**: LangChain's composable wrapper that applies any compressor on top of any base retriever.
- **Financial relevance**: earnings Q&A has many semantically similar passages (multiple margin figures, multiple ratios); reranking surfaces the one chunk that actually answers the question.

> **Note**: first run downloads the cross-encoder model (~80 MB). Subsequent runs use the local cache.

## Architecture

```
docs/*.txt
    │
    ▼
Ingest → Qdrant / Chroma vector store
    │
    ▼
Base Retriever (bi-encoder, k=20)  ← fast approximate search
    │
    ▼
CrossEncoderReranker (ms-marco-MiniLM-L-6-v2, top_n=4)  ← precise joint scoring
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

**Chroma fallback**: set `VECTOR_STORE=chroma` in `.env`.

## Run

```powershell
python ingest.py
streamlit run app.py
```

## How it works

1. `ingest.py` loads `.txt` files, splits into ~400-token chunks, embeds with OpenAI/Ollama, stores in Qdrant (or Chroma).
2. `app.py` builds a base retriever that fetches **k=20** candidates using the bi-encoder (fast ANN).
3. A `HuggingFaceCrossEncoder` (`ms-marco-MiniLM-L-6-v2`) scores all 20 `(query, chunk)` pairs jointly — this is the key precision step.
4. `CrossEncoderReranker` keeps the **top 4** highest-scoring chunks, discarding the rest.
5. The 4 chunks are formatted as context and passed to the LLM with a `ChatPromptTemplate`.
6. The Streamlit UI streams the answer with `st.write_stream`.

## When to use this pattern

Use reranking when your corpus has many topically similar documents (multiple earnings reports, multiple risk sections) where the bi-encoder cannot distinguish the best passage. Reranking adds latency but dramatically improves answer grounding — worth it for production RAG over large financial corpora.

## Tech stack

Qdrant / Chroma · HuggingFace cross-encoder (ms-marco-MiniLM-L-6-v2) · LangChain LCEL · OpenAI/Ollama · Streamlit
