"""Streamlit UI for Corrective RAG (CRAG)."""
import streamlit as st
from dotenv import load_dotenv
from graph import build_graph, MAX_RETRIES

load_dotenv()

st.set_page_config(page_title="10 · Corrective RAG", page_icon="🔄")
st.title("10 · Corrective RAG (CRAG)")
st.caption(
    "Grades retrieved docs for relevance; rewrites the query and retries "
    f"up to **{MAX_RETRIES}×** before generating an answer."
)

question = st.text_input(
    "Ask a financial question:",
    value="What is GlobalBank's cost-to-income ratio and capital position?",
)

if st.button("Run CRAG", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    graph = build_graph()

    # Stream graph execution step-by-step to capture the trace
    trace = []
    final_state = None

    progress = st.status("Running CRAG pipeline…", expanded=True)
    with progress:
        for step in graph.stream(
            {
                "question": question,
                "original_question": question,
                "documents": [],
                "generation": "",
                "retries": 0,
            }
        ):
            node_name, node_output = next(iter(step.items()))
            trace.append((node_name, node_output))
            st.write(f"✓ **{node_name}**")
        progress.update(label="Done!", state="complete")

    # Reconstruct final state from trace
    final_state = {}
    for _, output in trace:
        final_state.update(output)

    # ── Decision trace ───────────────────────────────────────────────────────
    st.subheader("Decision Trace")
    for node_name, output in trace:
        if node_name == "retrieve":
            with st.expander(f"🔍 retrieve — fetched {len(output.get('documents', []))} docs"):
                for i, doc in enumerate(output.get("documents", []), 1):
                    st.text(f"[{i}] {doc.page_content[:200]}…")

        elif node_name == "grade_documents":
            kept = len(output.get("documents", []))
            label = "✅" if kept > 0 else "❌"
            retries = final_state.get("retries", 0)
            with st.expander(f"{label} grade_documents — {kept} relevant docs kept"):
                if kept == 0 and retries < 2:
                    st.info("No relevant docs found → rewriting query")
                elif kept == 0:
                    st.warning("Retries exhausted — generating from empty context")
                else:
                    for i, doc in enumerate(output["documents"], 1):
                        st.markdown(f"**Doc {i}:** {doc.page_content[:200]}…")

        elif node_name == "transform_query":
            retry_num = output.get("retries", "?")
            with st.expander(f"🔄 transform_query — retry {retry_num}/{MAX_RETRIES}"):
                st.markdown(f"**Rewritten question:** {output.get('question', '')}")
                st.caption(
                    "💡 Web-search extension point: in production, this is where "
                    "you would add a web_search_node to fetch live data."
                )

        elif node_name == "generate":
            pass  # shown below as main output

    # ── Final answer ─────────────────────────────────────────────────────────
    st.subheader("💬 Answer")
    st.markdown(final_state.get("generation", "(no answer generated)"))
