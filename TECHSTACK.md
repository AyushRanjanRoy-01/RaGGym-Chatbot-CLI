# RAG Scaffold — Tech Stack Decisions

## Purpose of this document

This is not a tutorial. It records **what we chose, why we chose it, what we rejected, and what caveats you must know before extending the scaffold**. Read this before touching any layer.

---

## Project Context

**Goal:** A developer scaffold for a financial domain RAG chatbot.
- Designed to go on a Gen AI engineer resume (Indian market).
- Plain `.txt` documents, no images or charts.
- Developer should be able to clone and run in under 5 minutes.
- Every layer must be swappable via config, not code changes.

---

## Layer 1 — LLM

### Chosen: Ollama (default) + OpenAI (documented alternative)

**Why Ollama as default:**
- Zero API key friction. A developer cloning this scaffold can run it immediately without signing up anywhere or adding billing.
- Exposes an OpenAI-compatible REST API at `localhost:11434` — the same LangChain `ChatOpenAI` class works with both by changing `base_url`. Swap cost is one config line.
- Runs entirely offline. For financial documents this matters — no data leaves the machine.
- Handles model management (download, versioning) in one binary. `ollama pull llama3.2:3b` is the full setup.

**Default model: `llama3.2:3b`**
- Runs on CPU (important on a developer laptop with no GPU).
- 3B params = ~2GB RAM. Fast enough for a demo.
- Upgrade path: `mistral:7b` or `llama3:8b` for better quality if GPU is available.

**Why OpenAI is a co-equal, prominently documented alternative (not a footnote):**
- The Indian Gen AI job market runs on cloud LLM APIs — OpenAI, Anthropic, Azure OpenAI. Interviewers expect you to know API key management, token cost control, rate limit handling, and retry logic.
- For a live interview demo, GPT-4o-mini will respond in 1–2 seconds; llama3.2:3b on CPU will take 10–15 seconds. Latency is a demo killer.
- The `.env.example` shows both configurations, with OpenAI commented in clearly.

**What was rejected:**
- **HuggingFace Inference API** — free tier is throttled and unreliable for a demo. Local HuggingFace (`transformers` pipeline) requires CUDA setup that breaks on Windows.
- **Groq** — excellent latency (LPU hardware) but another API key dependency. Good as a third option once the scaffold is established.
- **Azure OpenAI** — enterprise credential complexity. Not appropriate for a personal scaffold.

**Honest caveat:**
On CPU, any local model above 3B will be noticeably slow. If you're demoing live, use OpenAI. If you're demoing offline or care about data privacy, Ollama.

---

## Layer 2 — Embeddings

### Chosen: `nomic-embed-text` via Ollama

**Why:**
- Runs inside Ollama — same runtime as the LLM. No second process, no second setup step.
- **8192 token context window.** This is the critical differentiator. Financial paragraphs are long. `all-MiniLM-L6-v2` (the most commonly cited free model) has a 256-token context — it will silently truncate any chunk longer than ~200 words. `nomic-embed-text` handles full paragraphs.
- 768-dimensional output. Memory cost: 3KB per vector. 100k document chunks = ~300MB. Manageable.
- Matches or beats `all-MiniLM` on MTEB benchmarks while handling 32x longer input.

**Why the embedding model choice is more critical than the LLM choice in RAG:**
Retrieval quality is the ceiling on answer quality. If the wrong chunks are fetched, the best LLM still gives a wrong answer. A mediocre LLM with perfect retrieval beats a great LLM with noisy retrieval. Invest here first.

**CRITICAL constraint:** The embedding model must be identical at ingest time and query time. Changing `EMBED_MODEL` in `.env` after ingestion requires a full re-ingest. The scaffold warns about this in the ingest CLI.

**What was rejected:**
- **`all-MiniLM-L6-v2`** — 256-token context. Will silently truncate financial paragraphs. Not acceptable.
- **OpenAI `text-embedding-3-small`** — excellent quality (1536 dims) but adds API cost and key dependency to the ingestion pipeline. Documented as upgrade path, not default.
- **`mxbai-embed-large`** — 512-token context. Same truncation problem as MiniLM.

**Upgrade path:** Set `EMBED_PROVIDER=openai` and `EMBED_MODEL=text-embedding-3-small` in `.env`. Re-run ingest. No code changes.

---

## Layer 3 — Vector Database

### Chosen: Qdrant (default) → ChromaDB (lightweight fallback)

Both are implemented behind a `BaseVectorStore` abstract class. The swap is `VECTOR_STORE=qdrant` in `.env`.

### Qdrant as default

**Why it's meaningfully better, not just "more enterprise":**
- Written in **Rust**. Handles concurrent reads/writes safely at scale.
- **Indexed payload fields** — filter conditions are evaluated during HNSW graph traversal, not after. O(log n) filtered search vs Chroma's O(n) scan.
- **Hybrid search** (dense + sparse/BM25) — critical for financial documents where exact terms like "EBITDA", "Tier-1 capital", "Basel III" must match precisely, not just semantically.
- **Memory-mapped storage** — vectors live on disk, OS page cache manages what fits in RAM. Handles billions of vectors without OOM.
- **Named vectors** — one document can have multiple embeddings (title, body, summary) searched independently.
- **Quantization** — scalar (float32 → int8, 4x compression) and binary (32x compression) with optional re-scoring for accuracy.

**Local mode (no Docker required):**
`QdrantClient(path="./vectorstore")` runs Qdrant in-process. Zero infra setup — identical friction to ChromaDB. Docker is only needed when you want the full server (auth, gRPC, remote access, Qdrant dashboard).

**ChromaDB as fallback:** Set `VECTOR_STORE=chroma` in `.env`. Useful for developers who want the absolute minimum setup and don't need filtered search or hybrid search.

**What was rejected:**
- **FAISS** — pure vector math, no metadata, no filtering, no persistence built-in. You'd rebuild ChromaDB manually on top of it. Not appropriate for a scaffold.
- **Pinecone** — managed cloud only. Requires account + API key before a developer can run the scaffold. Violates the zero-friction principle.
- **Weaviate** — complex Docker Compose setup. GraphQL query interface. Steeper learning curve with no meaningful benefit for this use case.
- **Milvus** — production-grade but heavyweight. Requires Docker Compose with multiple services. Overkill for a scaffold.

---

## Layer 4 — Orchestration

### Chosen: LangChain (LCEL) — with honest caveats

**Why LangChain:**
- Largest ecosystem of integrations. Every LLM, every vector DB, every embedding provider has a LangChain wrapper. For a scaffold where the developer will swap components, this matters.
- LCEL (LangChain Expression Language) uses Python's pipe operator `|` to compose explicit, readable chains. No hidden prompts, no black-box behavior.
- Native streaming (`chain.stream()`), async (`chain.ainvoke()`), and LangSmith tracing built into every LCEL chain.
- Indian market familiarity — appears in the majority of Gen AI job descriptions.

**The caveats you must know (interviewers will probe this):**
- LangChain has had **frequent breaking changes** between versions (0.1 → 0.2 → 0.3 were all breaking). Pin your version.
- The legacy chain classes (`RetrievalQA`, `ConversationalRetrievalChain`) are deprecated. Use LCEL. Showing legacy chains in an interview is a red flag.
- "LangChain makes easy things hard and hard things impossible" is a real community criticism. Over-abstraction makes debugging non-obvious. Write thin wrappers so the chain layer is replaceable.
- The scaffold uses LCEL only. No legacy imports.

**LCEL vs Legacy — the one thing to demonstrate in an interview:**
```python
# Legacy (don't use — hidden prompt, can't stream, can't trace)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# LCEL (what the scaffold uses — explicit, streamable, traceable)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
```

**LlamaIndex — why it was not chosen as default:**
LlamaIndex has better first-class abstractions for pure RAG data pipelines (indexing, retrieval, document hierarchy). For a scaffold that extends to agents, tools, and multi-step chains, LangChain wins ecosystem coverage. **If you fork this scaffold for a pure RAG-only system with no agent requirements, consider migrating the chain layer to LlamaIndex.**

---

## Layer 5 — Chunking

### Chosen: `RecursiveCharacterTextSplitter`

**How it works:**
Tries to split on `\n\n` (paragraph break) first, then `\n`, then space, then character. Result: chunks that respect natural text boundaries.

**Default parameters:**
- `chunk_size=1000` — approximately one long paragraph. Enough context for a financial concept to be explained fully.
- `chunk_overlap=200` — 20% overlap prevents a key sentence being cut at a chunk boundary from disappearing. The overlap ensures it appears complete in at least one chunk.

**Why this over alternatives:**
- `CharacterTextSplitter` — fixed size on a single separator. Splits mid-sentence regularly. Do not use.
- `TokenTextSplitter` — precise context window control but splits on token boundaries, not sentence boundaries. Use when you need exact token counts (e.g. fitting into a 4096-token context).
- `SemanticChunker` — embeds each sentence and splits where cosine similarity drops. Most accurate for mixed-topic documents. Too slow for the ingestion step of a scaffold (requires embedding model inference per sentence during ingestion).

**Critical caveat for financial documents:**
`RecursiveCharacterTextSplitter` works well for **narrative prose** (MD&A sections, risk factors, earnings commentary). It will **break tabular data** — a balance sheet or income statement split mid-row loses its column-row relationship. If your financial documents contain tables:
1. Pre-process to extract tables separately before chunking.
2. Or use `MarkdownHeaderTextSplitter` if the source is structured markdown.
3. Inspect your actual documents before committing to a chunking strategy.

---

## Layer 6 — UI

### Chosen: Streamlit

**Why:**
- Developer familiar, widely known in the Indian ML/AI community.
- `st.chat_message`, `st.chat_input`, `st.write_stream` cover everything a RAG chat UI needs natively.
- `st.session_state` manages chat history with zero boilerplate.
- `streamlit run app.py` is the entire launch command — no build step, no separate process.

**Considered but not chosen:**
- **Chainlit** — purpose-built for RAG UIs, has native source citation (`cl.Text`) and step visualization (`cl.Step`). Better RAG UX. Rejected in favour of Streamlit because developer familiarity is higher and the scaffold prioritises low friction over polish.
- **Gradio** — oriented toward model input/output demos, not chat history. `gr.ChatInterface` is one function call but less flexible to extend.

**What it is not:**
Streamlit is a demo/prototype UI. For production: FastAPI backend + React frontend. Out of scope for this scaffold.

---

## Layer 7 — Configuration

### Chosen: Pydantic `BaseSettings` + `.env` file

**Why:**
- **Type validation at startup.** Every config value has a declared type. If `CHUNK_SIZE=abc` appears in `.env`, the application crashes with a clear error at startup — not a silent bug mid-run.
- **Environment variable override.** Any field can be overridden: `VECTOR_STORE=qdrant python app.py` works with no code changes. This is how the swap-without-code-changes story works.
- **IDE autocomplete and static analysis.** `settings.chunk_size` is type-hinted everywhere. Typos in config key names are caught by the linter.
- **Single source of truth.** One `Settings` class, instantiated once as a singleton, imported everywhere. No scattered `os.getenv()` calls.

**Package note:** Use `pydantic-settings` (separate package in Pydantic v2), not `pydantic.BaseSettings` from v1. Pin: `pydantic-settings>=2.0`.

---

## Layer 8 — Logging

### Chosen: `structlog`

**Why structured logging over `print()` or stdlib `logging`:**
Every log line emits JSON (or colored key=value in dev mode). This means logs are grep-able and parseable:
```bash
# Find all queries where top retrieval score was below 0.5
cat app.log | jq 'select(.event=="retrieval_complete" and .top_score < 0.5)'
```

**What to log in a RAG pipeline (required for debugging):**
1. **Retrieval:** query text, number of chunks returned, top similarity score, latency
2. **Generation:** model name, input token count, output token count, latency
3. **Errors:** exception type, query that caused it, which layer failed

Without these logs you cannot answer "why did it give a wrong answer" — was retrieval bad (wrong chunks) or generation bad (hallucination despite correct context)?

---

## Layer 9 — Evaluation (RAGAS)

### Chosen: RAGAS framework — `eval/` folder

**This layer was missing from the initial design. It is not optional for a resume-quality scaffold.**

A RAG system without an evaluation layer has no way to measure if retrieval quality is improving or degrading when you change chunk size, swap embedding models, or adjust `top_k`. Without metrics, you are guessing.

**RAGAS metrics used:**
- **Faithfulness** — does the answer contain only information present in the retrieved context? Measures hallucination.
- **Answer Relevance** — is the answer on-topic for the question asked?
- **Context Precision** — are the retrieved chunks actually relevant to the question?
- **Context Recall** — did the retrieval surface all the relevant information that exists?

**Why this sets the scaffold apart:**
90% of RAG tutorials stop at "it works." An `eval/` folder with RAGAS wired up signals you understand that RAG is a system to be measured and tuned, not just assembled.

---

## Summary Table

| Layer | Default | Swap to | Trigger for swap |
|---|---|---|---|
| LLM | Ollama `llama3.2:3b` | OpenAI `gpt-4o-mini` | Live demo, production, API cost tracking |
| Embeddings | `nomic-embed-text` | `text-embedding-3-small` | Higher retrieval quality needed |
| Vector DB | Qdrant (local in-process) | ChromaDB | lightweight fallback only |
| Orchestration | LangChain LCEL | LlamaIndex | Pure RAG, no agents, complex indexing |
| Chunking | RecursiveCharacter | SemanticChunker | Mixed-topic docs, poor retrieval scores |
| UI | Streamlit | FastAPI + React | Production deployment |
| Config | Pydantic BaseSettings | — | Nothing better in Python today |
| Logging | structlog | — | Loguru if structlog feels heavy |
| Evaluation | RAGAS | — | — |
