"""
08 · SQL Agent
Text-to-SQL ReAct agent over a sample financial SQLite database.
Run `python setup_db.py` first to create financials.db.
"""
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


# ── Provider helper ──────────────────────────────────────────────────────────

def get_llm():
    if os.getenv("LLM_PROVIDER", "openai") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=os.getenv("LLM_MODEL", "llama3.2:3b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0,
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)


# ── Agent factory ────────────────────────────────────────────────────────────

@st.cache_resource
def build_agent():
    from langchain_community.utilities import SQLDatabase
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langgraph.prebuilt import create_react_agent

    db = SQLDatabase.from_uri("sqlite:///financials.db")
    toolkit = SQLDatabaseToolkit(db=db, llm=get_llm())

    system_prompt = (
        "You are a financial data analyst with read-only access to a SQLite database "
        "containing company and quarterly financial data.\n\n"
        "Tables:\n"
        "  - companies(id, name, ticker, sector)\n"
        "  - quarterly_financials(id, company_id, fiscal_quarter, revenue, net_income, ebitda)\n\n"
        "Guidelines:\n"
        "1. Always inspect the schema before writing SQL.\n"
        "2. Write SELECT-only queries — never INSERT, UPDATE, DELETE, or DROP.\n"
        "3. Use joins when company names are requested alongside financial figures.\n"
        "4. Present numbers in a readable format (e.g. $4.2M).\n"
        "5. If a query returns no rows, say so clearly."
    )

    # SQLDatabaseToolkit provides: sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker
    return create_react_agent(get_llm(), tools=toolkit.get_tools(), prompt=system_prompt)


# ── Streamlit UI ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="08 · SQL Agent", page_icon="🗄️")
st.title("08 · SQL Agent")
st.caption("Ask natural-language questions about the financial database — the agent writes and runs SQL.")

with st.sidebar:
    st.markdown("**Sample questions**")
    st.markdown(
        "- Which company had the highest revenue in Q4 2024?\n"
        "- Show total annual revenue for each company in 2024.\n"
        "- What is the net income margin for Acme Corp across all quarters?\n"
        "- List all companies in the Technology sector.\n"
        "- Compare EBITDA across companies for Q3 2024."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent = build_agent()
            result = agent.invoke({"messages": [("human", prompt)]})

            # Collect SQL tool calls for the expander
            tool_log = []
            for step in result.get("messages", []):
                cls = step.__class__.__name__
                if cls == "ToolMessage":
                    tool_log.append(
                        f"**Tool:** `{getattr(step, 'name', '?')}`\n```sql\n{step.content}\n```"
                    )

            if tool_log:
                with st.expander("🗄️ SQL tool calls", expanded=False):
                    st.markdown("\n\n".join(tool_log))

            final = result["messages"][-1].content
            st.markdown(final)
            st.session_state.messages.append({"role": "assistant", "content": final})

        except Exception as exc:
            msg = str(exc)
            if "no such table" in msg or "unable to open" in msg.lower():
                err = "Database not found. Run `python setup_db.py` first, then restart the app."
            else:
                err = f"Error: {msg}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
