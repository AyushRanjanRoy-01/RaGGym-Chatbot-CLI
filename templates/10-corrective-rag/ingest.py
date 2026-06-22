"""Ingest ./docs/*.txt into the vector store."""
import os
import glob
from dotenv import load_dotenv

load_dotenv()


def get_embeddings():
    if os.getenv("EMBED_PROVIDER", "openai") == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=os.getenv("EMBED_MODEL", "text-embedding-3-small"))


def main():
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = []
    for path in glob.glob("docs/*.txt"):
        docs.extend(TextLoader(path).load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = get_embeddings()

    if os.getenv("VECTOR_STORE", "qdrant") == "chroma":
        from langchain_chroma import Chroma
        Chroma.from_documents(chunks, embeddings, persist_directory="./vectorstore",
                              collection_name="financials")
    else:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        from langchain_qdrant import QdrantVectorStore
        # Derive dimension at runtime — never hardcode
        dim = len(embeddings.embed_query("test"))
        client = QdrantClient(path="./vectorstore")
        client.recreate_collection(
            collection_name="financials",
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        QdrantVectorStore.from_documents(
            chunks, embeddings, path="./vectorstore", collection_name="financials"
        )

    print(f"Ingested {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
