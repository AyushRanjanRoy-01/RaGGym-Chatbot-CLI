# 01 · Simple Financial Chatbot
Plain multi-turn LLM assistant with a financial-analyst persona — no retrieval, no vector store.

## What this demonstrates
- LCEL chain composition: `ChatPromptTemplate | LLM | StrOutputParser`
- `MessagesPlaceholder` for lightweight in-session memory (no DB required)
- Token-streaming to the UI via `st.write_stream` — critical for UX on long financial answers
- Provider-swappable design (OpenAI ↔ Ollama) without changing application logic

## Architecture
```
User input
    │
    ▼
ChatPromptTemplate
 (system + history + input)
    │
    ▼
LLM (OpenAI / Ollama)
    │
    ▼
StrOutputParser ──► st.write_stream ──► Streamlit UI
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
# set LLM_PROVIDER=ollama in .env
```

## Run
```powershell
streamlit run app.py
```

## How it works
1. On each submit, `st.session_state.history` (list of `HumanMessage`/`AIMessage`) is passed as `history`.
2. `ChatPromptTemplate` inserts the system persona, injects history via `MessagesPlaceholder`, then appends the new human turn.
3. The chain streams tokens through `StrOutputParser` directly into `st.write_stream`.
4. The assistant's full response is captured and appended to `history` for the next turn.

## When to use this pattern
- Baseline for any financial Q&A prototype before adding retrieval
- When your knowledge is already in the model (macro concepts, accounting definitions)
- Rapid demos where document ingestion infra isn't ready yet

## Tech stack
LangChain 0.3 LCEL · ChatOpenAI / ChatOllama · Streamlit
