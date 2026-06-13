import structlog
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import VectorStoreRetriever

log = structlog.get_logger()

_SYSTEM_PROMPT = """\
You are a precise assistant for financial document analysis.
Answer ONLY from the context provided. If the answer is not in the context, say:
"I don't have enough information in the provided documents to answer this."
Cite the source document name when relevant. Be concise.

Context:
{context}"""


def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


def _format_history(messages: list[dict]) -> list:
    result = []
    for msg in messages:
        if msg["role"] == "user":
            result.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            result.append(AIMessage(content=msg["content"]))
    return result


def build_rag_chain(retriever: VectorStoreRetriever):
    """
    Returns an LCEL chain. Input: {"question": str, "history": list[dict]}
    Output: str (streaming-compatible)
    """
    from rag.llm import get_llm

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | _format_docs,
            "question": RunnablePassthrough() | RunnableLambda(lambda x: x["question"]),
            "history": RunnableLambda(lambda x: _format_history(x.get("history", []))),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )

    log.info("rag_chain_built")
    return chain
