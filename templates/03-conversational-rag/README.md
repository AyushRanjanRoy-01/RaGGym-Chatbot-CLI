# 03 · Conversational RAG
History-aware retrieval-augmented generation — follow-up questions work correctly across turns.

## What this demonstrates
- Why naive RAG breaks on follow-ups: "what about the prior quarter?" sent verbatim to the retriever returns irrelevant chunks because the retriever has no memory
- `create_history_aware_retriever`: a sub-chain that rewrites ambiguous follow-ups into standalone questions *before* retrieval
- `create_retrieval_chain` + `create_stuff_documents_chain`: the current non-deprecated LCEL-based constructors (not ConversationalRetrievalChain)
- Two separate prompt templates — one for query rewriting, one for final answering — each with `MessagesPlaceholder` for history

## Architecture
```
User question + chat_history
        │
        ▼
  Contextualize prompt
  (rewrite follow-up → standalone)
        │
        ▼
  Retriever (k=4, Qdrant / Chroma)
        │
   top-4 chunks
        │
        ▼
  Answer prompt
  (context + history + question)
        │
        ▼
  LLM ──► streamed answer
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
python ingest.py          # index docs once
streamlit run app.py
```

## How it works
1. `ingest.py` splits and embeds `.txt` files from `./docs` into Qdrant (or Chroma) exactly as in template 02.
2. `build_rag_chain()` constructs two prompts: a *contextualize* prompt (rewrites follow-ups) and an *answer* prompt (cites sources), both with `MessagesPlaceholder("chat_history")`.
3. `create_history_aware_retriever(llm, retriever, contextualize_prompt)` wraps the retriever — when history is non-empty it first calls the LLM to rewrite the question, then retrieves.
4. `create_stuff_documents_chain(llm, answer_prompt)` formats retrieved docs and generates a cited answer.
5. `create_retrieval_chain` composes steps 3 & 4 into a single runnable that accepts `{input, chat_history}`.
6. The app manually iterates `rag_chain.stream(...)` and collects only the `"answer"` key from each dict chunk, updating a `st.empty()` placeholder in real time.

## When to use this pattern
- Multi-turn financial research sessions ("What was EPS? What about the prior quarter? How does that compare to guidance?")
- Any scenario where users naturally refer back to earlier answers with pronouns or implicit references
- Before upgrading to agentic RAG (tool-calling) — this is the cheapest history-aware step

## Tech stack
LangChain 0.3 LCEL · `create_history_aware_retriever` · Qdrant (local) · OpenAIEmbeddings / OllamaEmbeddings · Streamlit
