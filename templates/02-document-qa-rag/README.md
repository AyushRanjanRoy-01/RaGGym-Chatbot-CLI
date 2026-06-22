# 02 · Document Q&A RAG
Classic single-pass retrieval-augmented generation over financial documents.

## What this demonstrates
- Offline ingestion pipeline: load → chunk → embed → store (Qdrant local or Chroma)
- LCEL RAG chain with parallel context retrieval and question passthrough
- Source citation in answers — essential for audit trails in financial use cases
- Provider-swappable embeddings (OpenAI `text-embedding-3-small` ↔ Ollama `nomic-embed-text`)

## Architecture
```
[ingest.py]
  .txt files ──► TextLoader ──► RecursiveCharacterTextSplitter
                                        │
                                   Embeddings
                                        │
                               Qdrant / Chroma (local)

[app.py]
  Question ──► Retriever (k=4) ──► format_docs ──┐
  Question ─────────────────── passthrough ───────┤
                                                  ▼
                                          RAG Prompt + LLM ──► Answer (streamed)
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then fill in OPENAI_API_KEY
```
Ollama backup:
```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# set LLM_PROVIDER=ollama and EMBED_PROVIDER=ollama in .env
```

## Run
```powershell
python ingest.py          # index docs once (re-run after adding new files)
streamlit run app.py
```

## How it works
1. `ingest.py` loads all `.txt` files from `./docs`, splits into 1 000-token chunks (200 overlap), and stores embeddings in Qdrant at `./vectorstore`.
2. Embedding dimension is probed at runtime via `embed_query("test")` — no hardcoding needed.
3. `app.py` loads the same vector store at startup (`@st.cache_resource` avoids re-opening on each interaction).
4. On each question, the LCEL chain fetches the top-4 chunks in parallel with a `RunnablePassthrough` for the question.
5. `format_docs` prefixes each chunk with its `[source]` filename so the LLM can cite it.
6. The answer streams back via `st.write_stream`; history is stored in `st.session_state` for display only (this template is single-pass — see 03 for history-aware retrieval).

## When to use this pattern
- One-shot lookup over a fixed document set (earnings releases, 10-Ks, term sheets)
- When questions are self-contained and don't reference prior conversation turns
- Baseline before adding conversation history (template 03)

## Tech stack
LangChain 0.3 LCEL · Qdrant (local) · OpenAIEmbeddings / OllamaEmbeddings · Streamlit
