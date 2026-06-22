import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_embeddings():
    if os.getenv("EMBED_PROVIDER", "openai") == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
                                base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))


def main():
    docs_dir = Path("./docs")
    txt_files = list(docs_dir.rglob("*.txt"))
    if not txt_files:
        print("No .txt files found in ./docs — add documents and re-run.")
        return

    raw_docs = []
    for path in txt_files:
        loader = TextLoader(str(path), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.metadata["source"] = path.name
        raw_docs.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(raw_docs)
    print(f"Loaded {len(raw_docs)} document(s) → {len(chunks)} chunks")

    embeddings = get_embeddings()
    # Derive vector dimension from a live probe — avoids hardcoding per provider
    dim = len(embeddings.embed_query("test"))

    vector_store_type = os.getenv("VECTOR_STORE", "qdrant")

    if vector_store_type == "chroma":
        from langchain_chroma import Chroma
        Chroma.from_documents(chunks, embeddings, persist_directory="./vectorstore")
        print("Stored in Chroma at ./vectorstore")
    else:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore

        client = QdrantClient(path="./vectorstore")
        collection = "financial_docs"
        existing = [c.name for c in client.get_collections().collections]
        if collection not in existing:
            client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        QdrantVectorStore.from_documents(
            chunks, embeddings,
            client=client, collection_name=collection,
        )
        print(f"Stored {len(chunks)} chunks in Qdrant (local) at ./vectorstore")


if __name__ == "__main__":
    main()
