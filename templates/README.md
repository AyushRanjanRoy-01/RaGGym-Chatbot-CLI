# GenAI / RAG Template Gallery — Financial Domain

Ten self-contained, ready-to-run templates spanning the full progression from a plain chatbot to multi-agent and self-correcting RAG. Each folder is an independent mini-repo: copy it, install, run. Built for the financial-document use case (earnings, filings, structured financials).

> Every template defaults to **OpenAI `gpt-4o-mini`** with **Ollama `llama3.2:3b`** as a drop-in offline backup, and **Qdrant (local, no Docker)** as the vector store with **Chroma** as a fallback — all switchable via `.env`, zero code changes.

---

## The ten templates

| # | Template | Pattern | Framework | Level |
|---|----------|---------|-----------|-------|
| 01 | [simple-chatbot](01-simple-chatbot) | Streaming chat + memory, no retrieval | LCEL | Foundations |
| 02 | [document-qa-rag](02-document-qa-rag) | Classic single-pass RAG with citations | LCEL | Foundations |
| 03 | [conversational-rag](03-conversational-rag) | History-aware retrieval for follow-ups | LCEL | Production RAG |
| 04 | [hybrid-search-rag](04-hybrid-search-rag) | Dense + BM25 sparse fusion (Qdrant native) | LCEL + Qdrant | Production RAG |
| 05 | [reranking-rag](05-reranking-rag) | Retrieve-wide → cross-encoder rerank | LCEL | Advanced retrieval |
| 06 | [multi-query-rag](06-multi-query-rag) | RAG-Fusion: query expansion + RRF | LCEL | Advanced retrieval |
| 07 | [agentic-rag](07-agentic-rag) | ReAct agent: retrieve vs. compute, with tools | LangGraph | Agentic |
| 08 | [sql-agent](08-sql-agent) | Text-to-SQL over a financial database | LangGraph | Agentic |
| 09 | [multi-agent-analyst](09-multi-agent-analyst) | Researcher → Analyst → Writer pipeline | LangGraph | Multi-agent |
| 10 | [corrective-rag](10-corrective-rag) | CRAG: grade docs, self-correct, retry | LangGraph | Multi-agent |

---

## Shared conventions (identical across all ten)

**Providers** — set in each folder's `.env`:

| Concern | Default | Backup | Switch |
|---|---|---|---|
| LLM | OpenAI `gpt-4o-mini` | Ollama `llama3.2:3b` | `LLM_PROVIDER=ollama` |
| Embeddings | OpenAI `text-embedding-3-small` | Ollama `nomic-embed-text` | `EMBED_PROVIDER=ollama` |
| Vector store | Qdrant (local, in-process) | Chroma | `VECTOR_STORE=chroma` |

- The embedding **dimension is detected at runtime** (a probe `embed_query`), so swapping OpenAI ↔ Ollama needs no code change — just re-run ingest.
- Every RAG template ships a tiny `docs/sample_financials.txt` so it runs out-of-the-box before you add your own documents.
- Templates 04–06, 09–10 are advanced retrieval/agent patterns; 07–10 are agentic and use LangGraph.

---

## Quick start (any template)

```powershell
cd 02-document-qa-rag          # pick any folder
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env         # then paste your OPENAI_API_KEY

# RAG templates: ingest first
python ingest.py               # 08 uses: python setup_db.py
streamlit run app.py
```

**Offline / no API key?** Install [Ollama](https://ollama.com), then:
```powershell
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# in .env: LLM_PROVIDER=ollama  and  EMBED_PROVIDER=ollama
```

---

## Suggested learning / demo order

1. **01 → 02 → 03** — understand chat, retrieval, and conversational memory.
2. **04 → 05 → 06** — the three levers that fix bad retrieval (lexical match, reranking, query expansion).
3. **07 → 08** — when the LLM should *act* (call tools, query databases) instead of just answer.
4. **09 → 10** — orchestrating multiple steps and making the system self-correct.

## Skills this gallery demonstrates

`RAG` · `hybrid search` · `reranking` · `RAG-Fusion` · `ReAct agents` · `tool use` · `text-to-SQL` · `multi-agent orchestration` · `corrective/self-RAG` · `LangChain LCEL` · `LangGraph` · `Qdrant` · `Chroma` · `local + hosted LLMs` · `Streamlit`
