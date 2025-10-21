# backend/app/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.config.setting import APP_NAME, API_VERSION
from app.db.connection import get_db, init_db, engine, Base
from app.db.models import Document, Chunk
from app.core.embeddings import load_embedding_model
from app.core.retriever import retrieve_similar_chunks
from app.api.library import router as library_router
from app.api.upload import router as upload_router
from typing import List, Dict, Optional
from pydantic import BaseModel
from contextlib import asynccontextmanager

# -------------------------------------------------------------------
# ✅ LIFESPAN HANDLER (replaces deprecated on_event)
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Application startup: Initializing database and loading models...")
    await init_db()
    load_embedding_model()
    print("✅ Startup complete. Ready to accept requests.")

    # Yield control to FastAPI (runs your app)
    yield

    print("🧹 Application shutdown: Closing database engine...")
    await engine.dispose()
    print("✅ Shutdown complete.")

# -------------------------------------------------------------------
# ✅ FastAPI app instance
# -------------------------------------------------------------------
app = FastAPI(
    title=f"{APP_NAME} {API_VERSION}",
    description="A production-ready multimodal RAG system backend.",
    version=API_VERSION,
    lifespan=lifespan,  # 👈 NEW: replaced @app.on_event
)

# Register high-level library API routes (embed/chunk/store/retrieve/upsert)
app.include_router(library_router, prefix="/api")
# Register upload endpoints (file/json upload that triggers background processing)
app.include_router(upload_router, prefix="/api")

# -------------------------------------------------------------------
# ✅ Health check route
# -------------------------------------------------------------------
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok", "message": "Multimodal RAG System is running!"}

# -------------------------------------------------------------------
# ✅ Query schemas
# -------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    document_type: Optional[str] = None
    min_similarity: float = 0.7

class QueryResponse(BaseModel):
    query: str
    results: List[Dict]
    total_results: int

# -------------------------------------------------------------------
# ✅ Main API Endpoint
# -------------------------------------------------------------------
@app.post(
    "/api/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK
)
async def query_rag_system(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Performs a semantic search against the knowledge base using the provided query.
    """
    print(f"Received query request: {request.query}")
    try:
        results = await retrieve_similar_chunks(
            db=db,
            query_text=request.query,
            top_k=request.top_k,
            document_type=request.document_type,
            min_similarity=request.min_similarity
        )
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results)
        )
    except Exception as e:
        print(f"Error during query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the query: {e}"
        )
