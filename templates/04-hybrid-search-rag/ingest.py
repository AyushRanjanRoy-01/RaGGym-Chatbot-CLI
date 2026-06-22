import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse

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
    loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    sparse = FastEmbedSparse(model_name="Qdrant/bm25")  # BM25 sparse encoder via fastembed

    # QdrantVectorStore.from_documents handles collection creation with BOTH
    # dense + sparse vector configs when mode=HYBRID — simplest correct path
    QdrantVectorStore.from_documents(
        chunks,
        embedding=embeddings,
        sparse_embedding=sparse,
        path="./vectorstore",
        collection_name="financials_hybrid",
        vector_name="dense",
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    print(f"Ingested {len(chunks)} chunks into Qdrant hybrid collection.")


if __name__ == "__main__":
    main()
