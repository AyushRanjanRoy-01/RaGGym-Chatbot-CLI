import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
import structlog

from core.logging import setup_logging
from config import settings
from endpoints.rag_assistant_endpoint import router as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize robust structlog configuration (uvicorn-like console log with color matching settings.log_level)
    setup_logging()
    
    log = structlog.get_logger()
    log.info(
        "app_startup", 
        title=settings.app_title,
        log_level=settings.log_level,
        llm_provider=settings.llm_provider,
        embed_provider=settings.embed_provider,
        vector_store=settings.vector_store
    )
    yield
    log.info("app_shutdown")


app = FastAPI(
    title=settings.app_title,
    description=settings.app_description,
    lifespan=lifespan
)

app.include_router(rag_router, prefix="/api/v1")

if __name__ == "__main__":
    # We call setup_logging here to catch uvicorn internal logs on startup config
    setup_logging()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_config=None)
