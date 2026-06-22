"""Streamlit UI for the multi-agent financial analyst pipeline."""
import streamlit as st
from dotenv import load_dotenv
from graph import build_graph

load_dotenv()

st.set_page_config(page_title="09 · Multi-Agent Analyst", page_icon="🏦")
st.title("09 · Multi-Agent Financial Analyst")
st.caption(
    "Three specialized agents: **Researcher** → **Analyst** → **Writer** "
    "produce an inspectable financial brief."
)

question = st.text_input(
    "Enter a financial question or topic:",
    value="What are the key financial highlights and risks for ACME Financial Corp?",
)

if st.button("Generate Brief", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    graph = build_graph()

    with st.spinner("Running multi-agent pipeline…"):
        result = graph.invoke({"question": question})

    # ── Node 1: Researcher output ────────────────────────────────────────────
    with st.expander("🔍 Researcher — retrieved context", expanded=False):
        st.text(result["research"] or "(no documents retrieved)")

    # ── Node 2: Analyst output ───────────────────────────────────────────────
    with st.expander("📊 Analyst — key figures & risks", expanded=False):
        st.markdown(result["analysis"])

    # ── Node 3: Writer output (main output) ─────────────────────────────────
    st.subheader("📝 Final Report")
    st.markdown(result["report"])
