import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


def get_llm():
    if os.getenv("LLM_PROVIDER", "openai") == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=os.getenv("LLM_MODEL", "llama3.2:3b"),
                          base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"), temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)


SYSTEM_PROMPT = (
    "You are a knowledgeable financial assistant specialising in equity research, "
    "earnings analysis, and corporate finance. Provide concise, accurate answers. "
    "Always clarify when information may be outdated or when professional advice is warranted."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),   # injects prior turns for multi-turn context
    ("human", "{input}"),
])

chain = prompt | get_llm() | StrOutputParser()

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("Financial Assistant")
st.caption("Powered by LangChain LCEL · no retrieval, pure LLM reasoning")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

if user_input := st.chat_input("Ask a financial question…"):
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        # st.write_stream consumes the generator and renders tokens as they arrive
        response = st.write_stream(
            chain.stream({"input": user_input, "history": st.session_state.history})
        )

    st.session_state.history.extend([
        HumanMessage(content=user_input),
        AIMessage(content=response),
    ])
