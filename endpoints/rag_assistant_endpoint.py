from pydantic import BaseModel, Field
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pathlib import Path
import structlog

# Assume we reuse components correctly initialized elsewhere
from rag.embeddings import get_embeddings
from rag.vectorstore import get_vectorstore
from rag.chain import build_rag_chain
from rag.loaders import TextLoader
from rag.chunkers import RecursiveChunker

log = structlog.get_logger()
router = APIRouter()

APP_DIR = Path(__file__).resolve().parent.parent
DOCS_DIR = APP_DIR / "docs"

# --- Pydantic Models for Querying ---
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the sender (e.g., 'user', 'assistant')")
    content: str = Field(..., description="Content of the message")

class QueryRequest(BaseModel):
    question: str = Field(..., description="User's query")
    history: list[ChatMessage] = Field(default_factory=list, description="Chat context history")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="The assistant's generated response")

# --- Dependencies for Pipeline ---
def get_rag_chain():
    # In a production app, pipeline instances should be cached
    try:
        embeddings = get_embeddings()
        store = get_vectorstore(embeddings)
        retriever = store.as_retriever()
        return build_rag_chain(retriever)
    except Exception as e:
        log.error("pipeline_init_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Could not initialize RAG pipeline")

@router.post("/chat", response_model=QueryResponse, summary="Query the RAG Chatbot")
async def chat_endpoint(request: QueryRequest, chain=Depends(get_rag_chain)):
    log.debug("chat_request_received", question=request.question, history_len=len(request.history))
    
    try:
        # Note: chain.stream() is typically used for UI, but here we invoke for a direct JSON response.
        history_dict = [{"role": msg.role, "content": msg.content} for msg in request.history]
        response_text = chain.invoke({"question": request.question, "history": history_dict})
        
        log.info("chat_response_generated", question=request.question)
        return QueryResponse(answer=response_text)
    except Exception as e:
        log.error("chat_generation_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload", summary="Upload a .txt document")
async def upload_document(file: UploadFile = File(...)):
    log.debug("upload_request_received", filename=file.filename)
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    target_path = DOCS_DIR / file.filename
    
    # Save the file
    content = await file.read()
    target_path.write_bytes(content)
    log.info("file_saved", filename=file.filename, size_bytes=len(content))

    # Ingest the file into vectorstore
    try:
        documents = TextLoader().load(str(target_path))
        chunks = RecursiveChunker().chunk(documents)
        embeddings = get_embeddings()
        store = get_vectorstore(embeddings)
        store.add_documents(chunks)

        log.info("file_ingested", filename=file.filename, chunks=len(chunks))
        return {"filename": file.filename, "status": "success", "chunks_indexed": len(chunks)}
    except Exception as e:
        log.error("ingestion_failed", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to ingest document")
