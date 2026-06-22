# 🏋️ RAGGym

> An open-source **gym for RAG interview prep** — train your RAG muscle by *coding* and by *chatting* with a corpus of books.

Most people "learn RAG" by reading. RAGGym makes you **practice** it. Point it at a
book (it ships ready for a 484-page AI engineering handbook, and is built to grow
to many books), then either:

- 💬 **Chat mode** — ask questions and get cited, grounded answers from an
  advanced RAG pipeline (hybrid search, reranking, multi-query, self-correction).
  This is where you *see* good RAG technique in action.
- 🧠 **Practice mode** — RAGGym pulls a concept from the book, generates a RAG
  interview question or a coding exercise into your IDE, you solve it, and a
  reviewer agent grades your work against the source material.

> [!NOTE]
> **Status: Phase 0 (foundation).** The package, config, logging, CLI, tests, and
> CI are in place. Ingestion, retrieval, chat, and practice land in the phases on
> the [roadmap](#-roadmap).

---

## ✨ Why it's different

- **Learn by doing, not just reading** — the practice loop turns passive material into reps.
- **Advanced RAG, not toy RAG** — hybrid + rerank + multi-query + corrective (CRAG) self-correction.
- **Multimodal-ready ingestion** — handles code blocks, tables, glossary, and callout boxes faithfully; figure/diagram captioning is a pluggable stage.
- **Multi-book corpus** — per-book metadata and namespacing from day one.
- **Every layer is swappable via `.env`** — LLM, embeddings, vector store, chunking, retrieval — no code changes.
- **Measured, not vibes** — a RAGAS evaluation layer to track quality as you tune.

---

## 🏗️ Architecture

```
src/raggym/
├── config/        typed, validated settings (pydantic-settings)
├── core/          structured logging (structlog), shared types
├── ingestion/     PDF → typed elements → chunk → embed → vector store   [Phase 1]
├── retrieval/     hybrid · reranker · multi-query · router               [Phase 2]
├── agents/        LangGraph: chat_graph + practice_graph              [Phase 2-3]
├── llm/ embeddings/ vectorstore/   provider factories (ollama/openai/anthropic · qdrant/chroma)
├── practice/      exercise scaffolding + pytest grading + reviewer       [Phase 3]
├── eval/          RAGAS metrics                                          [Phase 4]
└── apps/          Streamlit chat UI                                      [Phase 2]
```

**Chat graph:** `router → retrieve → rerank → grade (self-correct) → cite & answer`
**Practice graph:** `tutor (pick concept) → exercise generator → reviewer/grader`

---

## 🚀 Quickstart

> Requires Python 3.11+. [Ollama](https://ollama.com) for the zero-API-key local
> default (or set an OpenAI/Anthropic key in `.env`).

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS/Linux:           source .venv/bin/activate

# 2. Install the package + dev tools
pip install -e ".[dev]"

# 3. Configure
cp .env.example .env        # Windows: copy .env.example .env

# 4. Verify the install
raggym version
raggym config               # prints resolved settings (secrets redacted)

# 5. Run tests
pytest
```

Local-model setup (default, no API key):

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Coming next (see roadmap): `raggym ingest`, `raggym chat`, `raggym practice`.

---

## 🧰 Tech stack

| Layer | Default | Swap |
|---|---|---|
| LLM | Ollama `llama3.2:3b` | OpenAI `gpt-4o-mini` · Anthropic Claude |
| Embeddings | `nomic-embed-text` | OpenAI `text-embedding-3-small` |
| Vector DB | Qdrant (local, no Docker) | ChromaDB |
| Orchestration | LangChain LCEL + LangGraph | — |
| PDF parsing | Docling (+ PyMuPDF) | — |
| UI | Streamlit | — |
| Config / Logging | pydantic-settings / structlog | — |
| Eval | RAGAS | — |

---

## 🗺️ Roadmap

- [x] **Phase 0** — repo foundation: packaging, config, logging, CLI, tests, CI
- [ ] **Phase 1** — multimodal-ready ingestion → Qdrant, inspectable
- [ ] **Phase 2** — retrieval engine + chat graph + Streamlit chat with citations
- [ ] **Phase 3** — practice mode: exercise generation + coding harness + reviewer
- [ ] **Phase 4** — RAGAS evaluation + docs + figure/diagram captioning

---

## 🤝 Contributing

Issues and PRs welcome. Set up pre-commit before your first commit:

```bash
pip install -e ".[dev]"
pre-commit install
```

## 📄 License

[MIT](LICENSE)
