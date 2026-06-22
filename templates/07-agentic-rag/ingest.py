"""
ingest.py — Load ./docs/*.txt, chunk, embed, store in Qdrant (or Chroma).
Run once before starting app.py.
"""
import os
import glob
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_embeddings():
    if os.getenv("EMBED_PROVIDER", "openai") == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))


def ingest():
    txt_files = glob.glob("./docs/*.txt")
    if not txt_files:
        print("No .txt files found in ./docs/")
        return

    docs = []
    for path in txt_files:
        loader = TextLoader(path, encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = os.path.basename(path)
        docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} doc(s) → {len(chunks)} chunks")

    embeddings = get_embeddings()
    # Probe embedding dimension dynamically — never hardcode
    dim = len(embeddings.embed_query("test"))
    print(f"Embedding dimension: {dim}")

    vector_store = os.getenv("VECTOR_STORE", "qdrant")

    if vector_store == "chroma":
        from langchain_chroma import Chroma
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./vectorstore_chroma",
            collection_name="financial_docs",
        )
        print("Stored in Chroma (./vectorstore_chroma)")
    else:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore

        client = QdrantClient(path="./vectorstore")
        # Recreate collection so re-runs are idempotent
        client.recreate_collection(
            collection_name="financial_docs",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name="financial_docs",
            url=None,
            client=client,
        )
        print("Stored in Qdrant (./vectorstore)")


if __name__ == "__main__":
    ingest()
