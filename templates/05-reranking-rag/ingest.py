import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

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


def get_vector_store(chunks, embeddings):
    if os.getenv("VECTOR_STORE", "qdrant") == "chroma":
        from langchain_chroma import Chroma
        return Chroma.from_documents(chunks, embeddings, persist_directory="./vectorstore")
    from langchain_qdrant import QdrantVectorStore
    return QdrantVectorStore.from_documents(
        chunks, embeddings, path="./vectorstore", collection_name="financials_rerank"
    )


def main():
    loader = DirectoryLoader("./docs", glob="**/*.txt", loader_cls=TextLoader)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    get_vector_store(chunks, embeddings)
    print(f"Ingested {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
