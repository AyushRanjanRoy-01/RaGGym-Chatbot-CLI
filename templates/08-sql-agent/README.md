# 08 · SQL Agent

A Text-to-SQL ReAct agent that translates natural-language financial questions into SQL, runs them against a local SQLite database, and returns plain-English answers.

## What this demonstrates

- **Text-to-SQL for structured financial data**: The right tool when your data is tabular (companies, quarters, revenue) rather than prose documents.
- **Automatic schema inspection**: The agent calls `sql_db_schema` and `sql_db_list_tables` before writing queries — it discovers the schema at runtime, so no hand-holding is needed.
- **Read-only guardrails**: The system prompt instructs the agent to use SELECT-only queries; the LangChain SQL toolkit further limits exposure by not exposing DDL tools.
- **When to use SQL vs RAG**: RAG retrieves facts from unstructured text; SQL agents answer aggregation, filtering, and join queries over structured relational data with precision RAG cannot match.

## Architecture

```
User question (natural language)
        │
        ▼
  ReAct Agent  (create_react_agent + SQLDatabaseToolkit)
        │
        ├──► sql_db_list_tables   — discover available tables
        ├──► sql_db_schema        — inspect column definitions
        ├──► sql_db_query_checker — validate SQL before execution
        └──► sql_db_query         — execute SELECT → return results
                                          │
                                          ▼
                                   financials.db  (SQLite)
                                   ├── companies
                                   └── quarterly_financials
```

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env and set OPENAI_API_KEY

# 4. Create and seed the database
python setup_db.py
```

**Ollama backup** (no OpenAI key required):
```powershell
ollama pull llama3.2:3b
# In .env set LLM_PROVIDER=ollama
```

## Run

```powershell
streamlit run app.py
```

> **Note**: `setup_db.py` must be run before `streamlit run app.py`.

## How it works

1. `setup_db.py` creates `financials.db` with two tables: `companies` (3 rows) and `quarterly_financials` (12 rows across 3 companies × 4 quarters).
2. `SQLDatabase.from_uri("sqlite:///financials.db")` wraps the database; `SQLDatabaseToolkit` exposes four tools to the agent.
3. `create_react_agent` runs the ReAct loop: the LLM inspects the schema, drafts SQL, validates it, executes it, and formats the result into a plain-English answer.
4. Generated SQL and raw query results are shown in a collapsible expander for transparency.
5. A system prompt enforces SELECT-only intent and instructs the agent to inspect the schema before writing queries.

## Sample questions

- "Which company had the highest revenue in Q4 2024?"
- "Show total annual revenue for each company in 2024."
- "What is the net income margin for Acme Corp across all quarters?"
- "Compare EBITDA across companies for Q3 2024."

## When to use this pattern

- Your financial data lives in tables (revenue, P&L, positions, trades) — not in PDF reports.
- You need **aggregations, joins, and filters** that RAG cannot reliably produce.
- You want **exact answers** from a single source of truth rather than similarity-search approximations.
- Natural extension: swap SQLite for Postgres/Snowflake, add row-level security, or expose only specific views to the agent.

## Tech stack

LangChain 0.3 · LangGraph 0.2 (`create_react_agent`) · SQLDatabaseToolkit · SQLite · OpenAI / Ollama · Streamlit
