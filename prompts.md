given the multimodal rag system i want to build, i am already sending some dummy dara to the vector db, but i want you to make it in such a way that it wll be like a high level libray where users will have to work  high level code at the level of apis and send data to the pgvector which should be of type json and should also be able to retrieve that data, base on some similarity:# backend/app/main.py
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.config.setting import APP_NAME, API_VERSION
from app.db.connection import get_db, init_db, engine, Base
from app.db.models import Document, Chunk
from app.core.embeddings import load_embedding_model
from app.core.retriever import retrieve_similar_chunks
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
        )"



# Module 1: Embedding Generator

# Purpose: Convert any text to vector embeddings

# File: core/embeddings.py

# Key Functions:
# ________________________________________________________________________


# generate_embedding(text: str) → List[float]
# batch_generate_embeddings(texts: List[str]) → List[List[float]]
# Model: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions, free)


# backend/app/core/embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import numpy as np # Used by sentence-transformers, good to explicitly import
from app.config.setting import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

# --- Global Embedding Model Instance ---
# We load the model once when the application starts to avoid redundant loading.
# This saves memory and significantly speeds up embedding generation.
embedding_model: Optional[SentenceTransformer] = None

async def load_embedding_model():
    """
    Loads the SentenceTransformer model. This should be called once at application startup.
    """
    global embedding_model
    if embedding_model is None:
        try:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
            embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded successfully.")
            # Optional: Basic verification of the model's output dimension
            sample_embedding = embedding_model.encode("test string")
            if len(sample_embedding) != EMBEDDING_DIMENSION:
                print(f"WARNING: Model output dimension ({len(sample_embedding)}) "
                      f"does not match expected ({EMBEDDING_DIMENSION}) for {EMBEDDING_MODEL_NAME}.")
        except Exception as e:
            print(f"ERROR: Failed to load embedding model {EMBEDDING_MODEL_NAME}: {e}")
            embedding_model = None # Ensure it's None if loading failed

def get_embedding_model() -> SentenceTransformer:
    
    """
    Returns the loaded embedding model instance. Ensures it's loaded first.
    
    """
    if embedding_model is None:
        load_embedding_model() # Attempt to load if not already loaded
        if embedding_model is None: # If still None, loading failed
            raise RuntimeError("Embedding model is not loaded and could not be initialized.")
    return embedding_model

def generate_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for a single piece of text using the loaded model.
    """
    model = get_embedding_model()
    # model.encode returns a numpy array, convert to a Python list for database storage.
    embedding = model.encode(text).tolist()
    return embedding

def batch_generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generates vector embeddings for a list of texts in a batch.
    Batch processing is generally more efficient for sentence-transformers.
    """
    model = get_embedding_model()
    embeddings = model.encode(texts).tolist()
    return embeddings

#  --- Test function for local verification ---
if __name__ == "__main__":
    print("--- Testing Embedding Service ---")
    load_embedding_model() # Manually load the model for script execution

    test_text1 = "The quick brown fox jumps over the lazy dog."
    test_text2 = "A fast, reddish-brown canine leaps across a lethargic canine."
    test_text3 = "A cat sat on the mat."

    print(f"\nGenerating embedding for: '{test_text1}'")
    emb1 = generate_embedding(test_text1)
    print(f"Embedding length: {len(emb1)}")
    print(f"First 5 dimensions: {emb1[:5]}...")

    print(f"\nGenerating embeddings for multiple texts (batch processing):")
    batch_embeddings = batch_generate_embeddings([test_text1, test_text2, test_text3])
    print(f"Number of embeddings in batch: {len(batch_embeddings)}")
    print(f"Length of first batch embedding: {len(batch_embeddings[0])}")

    # Calculate cosine similarity manually for demonstration.
    # In pgvector, we'll use the '<=>' operator.
    def cosine_similarity(vec1, vec2):
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        return np.dot(vec1_np, vec2_np) / (np.linalg.norm(vec1_np) * np.linalg.norm(vec2_np))

    similarity_1_2 = cosine_similarity(emb1, batch_embeddings[1])
    similarity_1_3 = cosine_similarity(emb1, batch_embeddings[2])

    print(f"\nCosine Similarity (manually calculated):")
    print(f"'{test_text1}' vs '{test_text2}': {similarity_1_2:.4f} (Expected high - similar meaning)")
    print(f"'{test_text1}' vs '{test_text3}': {similarity_1_3:.4f} (Expected low - different meaning)")
    print("\n--- Embedding Service Test Complete ---")

# Module 6: Retriever
# Purpose: Search vector database for relevant chunks
# _____________________________________________________________________
# File: core/retriever.py

from typing import List, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.db.models import Chunk, Document
from app.core.embeddings import generate_embedding
from app.config.setting import EMBEDDING_DIMENSION


async def retrieve_similar_chunks(
    db: AsyncSession,
    query_text: str,
    top_k: int = 5,
    document_type: Optional[str] = None,
    min_similarity: float = 0.7
) -> List[Dict]:
    """
    Performs a vector similarity search in the database to find relevant chunks.

    Args:
        db: SQLAlchemy AsyncSession for DB interaction.
        query_text: Natural language user query.
        top_k: Number of top similar chunks to retrieve.
        document_type: Optional filter for document type.
        min_similarity: Minimum similarity threshold for returned chunks.

    Returns:
        A list of dictionaries representing relevant chunks with metadata.
    """
    print(f"🔍 Retrieving top {top_k} chunks for query: '{query_text}'")

    # 1. Generate an embedding for the query text
    query_embedding = generate_embedding(query_text)

    # ✅ Convert embedding list → Postgres vector string format
    query_vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    # 2. Build query filters dynamically
    where_clauses = []
    params = {
        "query_embedding": query_vector_str,
        "top_k": top_k,
    }

    if document_type:
        where_clauses.append("d.document_type = :document_type")
        params["document_type"] = document_type

    # 3. Construct SQL query
    # NOTE:
    # - We use `CAST(:query_embedding AS vector)` to ensure correct pgvector type casting
    # - We renamed `meta_data` → `metadata` to match standard model naming
    sql_query = f"""
SELECT
    c.id AS chunk_id,
    c.content,
    c.meta_data AS chunk_metadata,       -- ✅ fixed
    c.document_id,
    d.title AS document_title,
    d.document_type,
    d.meta_data AS document_metadata,    -- ✅ fixed
    1 - (c.embedding <=> CAST(:query_embedding AS vector)) AS similarity
FROM chunks c
JOIN documents d ON c.document_id = d.id
{"WHERE " + " AND ".join(where_clauses) if where_clauses else ""}
ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
LIMIT :top_k;
"""

    # 4. Execute query
    result = await db.execute(text(sql_query), params)
    retrieved_chunks_raw = result.fetchall()

    # 5. Structure and filter results
    retrieved_results = []
    for row in retrieved_chunks_raw:
        similarity = float(row.similarity)
        if similarity >= min_similarity:
            retrieved_results.append({
                "chunk_id": str(row.chunk_id),
                "content": row.content,
                "chunk_metadata": row.chunk_metadata,
                "document_id": str(row.document_id),
                "document_title": row.document_title,
                "document_type": row.document_type,
                "document_metadata": row.document_metadata,
                "similarity": similarity
            })

    print(f"✅ Retrieved {len(retrieved_results)} chunks "
          f"(out of {len(retrieved_chunks_raw)} before filtering).")
    return retrieved_results


# -------------------------------------------------------------------
# ✅ Local Testing (Optional)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    import uuid
    from app.db.connection import init_db, AsyncSessionLocal
    from app.db.models import Document, Chunk
    from app.core.embeddings import load_embedding_model

    async def test_retriever_functionality():
        print("\n--- Testing Retriever Functionality ---")
        await load_embedding_model()
        await init_db()

        # Insert sample data
        async with AsyncSessionLocal() as session:
            print("\n1️⃣ Inserting dummy data...")

            doc1 = Document(
                id=uuid.uuid4(),
                document_type="text",
                title="Medical Research Paper on Diabetes",
                file_path="/path/to/diabetes_paper.pdf",
                metadata={"author": "Dr. Smith", "year": 2023}
            )
            await session.merge(doc1)

            chunk_texts = [
                "Diabetes mellitus is a chronic metabolic disease characterized by elevated blood glucose levels.",
                "Symptoms of diabetes include frequent urination, increased thirst, and unexplained weight loss.",
                "Type 2 diabetes is managed through diet, exercise, and medications like metformin."
            ]

            for i, content in enumerate(chunk_texts):
                chunk = Chunk(
                    id=uuid.uuid4(),
                    document_id=doc1.id,
                    content=content,
                    chunk_index=i,
                    embedding=generate_embedding(content),
                    metadata={"page": i + 1}
                )
                await session.merge(chunk)

            doc2 = Document(
                id=uuid.uuid4(),
                document_type="image",
                title="X-ray Scan of a Lung",
                file_path="/path/to/lung_xray.png",
                metadata={"patient_id": "P123", "scan_date": "2024-01-15"}
            )
            await session.merge(doc2)

            chunk2 = Chunk(
                id=uuid.uuid4(),
                document_id=doc2.id,
                content="This X-ray image shows clear lungs with no signs of pneumonia.",
                chunk_index=0,
                embedding=generate_embedding("This X-ray image shows clear lungs with no signs of pneumonia."),
                metadata={"description_source": "AI_vision"}
            )
            await session.merge(chunk2)
            await session.commit()

            print("✅ Dummy data inserted successfully.")

        # Run queries
        async with AsyncSessionLocal() as session:
            print("\n2️⃣ Query: 'What are the signs of high blood sugar?'")
            query_results = await retrieve_similar_chunks(
                db=session,
                query_text="What are the signs of high blood sugar?",
                top_k=3
            )
            for i, res in enumerate(query_results):
                print(f"   --- Result {i+1} (Similarity: {res['similarity']:.4f}) ---")
                print(f"      Document: {res['document_title']} ({res['document_type']})")
                print(f"      Content: {res['content'][:120]}...")

        async with AsyncSessionLocal() as session:
            print("\n3️⃣ Query (filtered): 'Describe the X-ray results.' [image docs only]")
            results = await retrieve_similar_chunks(
                db=session,
                query_text="Describe the X-ray results.",
                top_k=2,
                document_type="image"
            )
            for i, res in enumerate(results):
                print(f"   --- Result {i+1} (Similarity: {res['similarity']:.4f}) ---")
                print(f"      Document: {res['document_title']} ({res['document_type']})")
                print(f"      Content: {res['content'][:120]}...")

        print("\n✅ Retriever test complete.")

    asyncio.run(test_retriever_functionality())


# backend/app/config/settings.py
from dotenv import load_dotenv
import os

# Load environment variables from .env file.
# This makes sure that when our app runs, it can pick up DATABASE_URL etc.
load_dotenv()

# --- Database Settings ---
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/rag_db")

# Ensure DATABASE_URL is properly configured
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

print(f"Configured database URL: {DATABASE_URL}")
    # In production, you might want to raise an exception here:
    # raise ValueError("DATABASE_URL environment variable is not set or contains placeholder.")

# --- Embedding Model Settings ---
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384 # This is the specific output dimension for 'all-MiniLM-L6-v2'

# --- Chunking Settings ---
TEXT_CHUNK_SIZE: int = 500
TEXT_CHUNK_OVERLAP: int = 50

# --- API Settings ---
APP_NAME: str = "Multimodal RAG System"
API_VERSION: str = "v1"

# backend/app/db/models.py
from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID, ARRAY # Specific PostgreSQL types
from sqlalchemy.orm import relationship # For defining relationships between models
from sqlalchemy.sql import func # For database functions like NOW()
from app.db.base import Base # Our custom Base class from base.py
from app.config.setting import EMBEDDING_DIMENSION # To get the embedding vector size
import uuid # For generating UUIDs

# --- Custom SQLAlchemy Type for pgvector's VECTOR ---
# SQLAlchemy doesn't natively know about PostgreSQL's 'VECTOR' type from pgvector.
# We need to teach it how to handle it.
class Vector(ARRAY):
    __qualname__ = 'Vector' # Helps with debugging and introspection

    def __init__(self, dim: int = EMBEDDING_DIMENSION, **kw):
        # We inherit from ARRAY to leverage its list-like behavior,
        # but override its SQL type definition.
        super().__init__(Integer(), dimensions=1, **kw) # Integer() is a placeholder base type
        self.dim = dim # Store the dimension for the VECTOR type

    # This method tells SQLAlchemy what SQL type to generate for the column
    # when Base.metadata.create_all() is called.
    def get_col_spec(self, **kw):
        return f"VECTOR({self.dim})"
    
    # This method defines how Python data (List[float]) is converted
    # to a string format that pgvector expects when INSERTING/UPDATING.
    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return value
            # Convert list of floats (e.g., [0.1, 0.2, 0.3])
            # into a string like "[0.1,0.2,0.3]"
            return "[" + ",".join(map(str, value)) + "]"
        return process

    # This method defines how the database's string representation of a vector
    # is converted back into a Python List[float] when SELECTING data.
    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            # pgvector might return "{0.1,0.2,0.3}" or "[0.1,0.2,0.3]"
            if isinstance(value, str):
                # Remove braces/brackets and split by comma, then convert to float
                return [float(x) for x in value.strip('{}[]').split(',') if x.strip()]
            return value
        return process


# --- Document Model ---
class Document(Base):
    """
    Represents a single source document uploaded to our RAG system.
    This could be a PDF file, an image, an audio file, or a video.
    It stores metadata about the original file.
    """
    __tablename__ = "documents" # The actual table name in our PostgreSQL database

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_type = Column(String(20), nullable=False) # e.g., 'text', 'image', 'audio', 'video'
    title = Column(String(255), nullable=True) # User-provided or extracted title
    file_path = Column(Text, nullable=True) # Path/URL to the original file (e.g., S3 URL, Base64 for small files)
    meta_data = Column(JSON, nullable=True) # Flexible JSONB column for extra metadata (e.g., author, source_url)
    created_at = Column(DateTime(timezone=True), server_default=func.now()) # Automatically set on creation
    updated_at = Column(DateTime(timezone=True), onupdate=func.now()) # Automatically updated on modification

    # Relationship to Chunk: One Document can have many Chunks.
    # 'cascade="all, delete-orphan"' means if a Document is deleted, all its associated Chunks are also deleted.
    # 'back_populates="document"' links this relationship to the 'document' attribute in the Chunk model.
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id='{self.id}', title='{self.title}', type='{self.document_type}')>"


# --- Chunk Model ---
class Chunk(Base):
    """
    Represents a smaller, processable piece (chunk) of a Document.
    Each chunk has its own embedding, allowing for granular semantic search.
    """
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Foreign Key linking back to the Document.
    # 'ondelete="CASCADE"' means if the parent Document is deleted, this Chunk is also deleted.
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    content = Column(Text, nullable=False) # The actual textual content of this chunk (e.g., text from a PDF page, image description)
    chunk_index = Column(Integer, nullable=True) # The order of this chunk within its parent document
    embedding = Column(Vector(EMBEDDING_DIMENSION), nullable=True) # Our custom Vector type for the pgvector column!
    meta_data = Column(JSON, nullable=True) # Chunk-specific metadata (e.g., page number, timestamp in video)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship to Document: Many Chunks belong to one Document.
    # 'back_populates="chunks"' links this relationship to the 'chunks' attribute in the Document model.
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<Chunk(id='{self.id}', document_id='{self.document_id}', index={self.chunk_index})>"


# --- Query Model (Optional for Analytics) ---
class Query(Base):
    """
    Stores records of user queries and the system's responses for analytics, logging,
    and potentially for feedback/reinforcement learning.
    """ 
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

# backend/app/db/connection.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config.setting import DATABASE_URL
import asyncio # For running async functions in a sync context (for init_db)
import sqlalchemy
from sqlalchemy import text  # Import text for raw SQL queries
from app.db import models  # Ensure all models are registered with Base.metadata

# --- 1. The Database Engine ---
# create_async_engine: This is our connection manager to the database.
# DATABASE_URL: The connection string from our settings.
# echo=False: If True, SQLAlchemy will print all SQL it executes to the console.
#             Useful for debugging, but verbose for production.
engine = create_async_engine(DATABASE_URL, echo=True)  # Enable SQL logging

# Import Base and models
from app.db.base import Base
from app.db.models import Document, Chunk, Query  # Import all models to ensure they are registered

# --- 3. Async Session Local Factory ---
# sessionmaker: This is a factory that will produce new AsyncSession objects.
# Each session represents a conversation with the database.
# autocommit=False, autoflush=False: Gives us explicit control over when to commit changes.
# bind=engine: Connects this session factory to our database engine.
# class_=AsyncSession: Specifies that the sessions produced will be asynchronous.
AsyncSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine, class_=AsyncSession
)

# --- 4. FastAPI Dependency for Database Session ---
# This is a special function designed for FastAPI's dependency injection system.
# When an API endpoint needs a database session, it will ask for 'db: AsyncSession = Depends(get_db)'.
# FastAPI will then call this function, manage the session's lifecycle (creation, yielding, closing).
async def get_db():
    db = AsyncSessionLocal() # Create a new asynchronous database session
    try:
        yield db # 'yield' means this function is a context manager.
                 # The session 'db' is provided to the calling function.
    finally:
        await db.close() # Ensure the session is closed after the request is processed,
                         # releasing the database connection.

# --- 5. Database Initialization Function ---
# This function will create our database tables and ensure the 'vector' extension is enabled.
async def init_db():
    async with engine.begin() as conn: # Get an asynchronous connection and start a transaction.
                                      # 'begin()' ensures all operations are atomic.
        print("\n--- Initializing Database Schema ---")

        # Debug: Print the DATABASE_URL to confirm it's loaded correctly
        print(f"Using DATABASE_URL: {DATABASE_URL}")

        # Test the database connection
        print("Testing database connection...")
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1;"))
            print("Database connection successful.")
        except Exception as e:
            print(f"ERROR: Could not connect to the database. {e}")
            return

        # Ensure connection remains open for 'vector' extension creation
        print("1. Ensuring 'vector' extension exists...")
        try:
            async with engine.begin() as vector_conn:
                await vector_conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            print("   'vector' extension enabled successfully (or already existed).")
        except Exception as e:
            print(f"   WARNING: Could not create 'vector' extension. Error: {e}")
            print("   Skipping 'vector' extension creation.")

        # Log table creation
        print("2. Creating tables based on models...")
        try:
            async with engine.begin() as table_conn:
                await table_conn.run_sync(Base.metadata.create_all)
                print("   Database tables created/updated successfully.")
        except Exception as e:
            print(f"   ERROR: Failed to create tables. Error: {e}")
            print("   Please check your model definitions and database connection.")

        # Ensure connection is open for table verification
        print("3. Verifying table creation...")
        try:
            async with engine.begin() as verify_conn:
                result = await verify_conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
                tables = [row[0] for row in result]
                print(f"   Tables in the database: {tables}")
        except Exception as e:
            print(f"   WARNING: Could not verify tables. Error: {e}")
        
        print("--- Database Schema Initialization Complete ---\n")

# --- 6. Direct Script Execution (for easy setup) ---
# This block allows you to run 'python app/db/connection.py' directly from your terminal
# to set up your database without starting the full FastAPI application.
if __name__ == "__main__":
    print("Attempting to initialize database via direct script execution...")
    asyncio.run(init_db()) # Run the async init_db function
    print("Database initialization script finished.")
    query_text = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    retrieved_chunk_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True) # A list of UUIDs of chunks used for the response
    user_id = Column(String(255), nullable=True) # Optional: ID of the user who made the query
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Query(id='{self.id}', text='{self.query_text[:50]}...')>"


eg=>"{
  "success": true,
  "data": {
    "internships": [
      {
        "id": "0710a1e4-946a-483d-98d6-274a7f9abd56",
        "title": "Another Test",
        "description": "No description available",
        "company": "LearnCode",
        "location": "location ",
        "type": "onsite",
        "category": "Marketing",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.725Z"
      },
      {
        "id": "fe392e49-73fd-4a32-a2c7-d72230312432",
        "title": "Testing notifiaction",
        "description": "No description available",
        "company": "LearnCode",
        "location": "Seed headquarter",
        "type": "remote",
        "category": "Data Science",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "1fb63ee6-8d11-405e-a1c0-f3a1487f7d74",
        "title": "Global Notification Test",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "Remote",
        "type": "Full-time",
        "category": "Technology",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "6a7a76bc-3f42-41b1-8854-fa05f4d3ded4",
        "title": "WordPress",
        "description": "No description available",
        "company": "SEEDINC",
        "location": "mile 6 nkwen",
        "type": "onsite",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "931c0ed9-bf8b-4d6a-9586-15c8e6ab77be",
        "title": "Backend Development",
        "description": "No description available",
        "company": "SEEDINC",
        "location": "Mile 6 Nkwen",
        "type": "onsite",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "f65b0c15-1a1b-497e-90f9-ba457bbd89fc",
        "title": "Data Annotation Intern",
        "description": "No description available",
        "company": "Seed Company",
        "location": "Bengaluru, Karnataka, India",
        "type": "onsite",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "1912e8d3-32b2-4ade-99ca-03916f69aa9d",
        "title": "Machine Learning Intern",
        "description": "No description available",
        "company": "Seed Company",
        "location": "San Francisco, CA, USA",
        "type": "onsite",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "9455ab20-e98d-48b9-a15f-3d9090a83d34",
        "title": "jjjhshheyey",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "johnnnbbd",
        "type": "onsite",
        "category": "Marketing",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "83e06d9a-afa9-417d-a771-2569d03f7789",
        "title": "foood",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "buda",
        "type": "onsite",
        "category": "Data Science",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "55e759e9-367d-4273-9f1e-14281f3142fb",
        "title": "weeeeejejeiekie",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "looooffnf",
        "type": "onsite",
        "category": "Data Science",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.726Z"
      },
      {
        "id": "bf843333-9756-4a1c-b6ce-94e668495799",
        "title": "Trigger Test Internship",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "Remote",
        "type": "Full-time",
        "category": "Technology",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "0eb2ae85-8562-41aa-bcfc-3630fe0dc742",
        "title": "Automatic Notification Test",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "Remote",
        "type": "Full-time",
        "category": "Technology",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "2c556059-13be-4971-9d16-279449be2b84",
        "title": "Automatic Notification Test",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "Remote",
        "type": "Full-time",
        "category": "Technology",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "6ff12c83-20de-47f9-b01f-1bf1cf9b98c2",
        "title": "ZIGEX program",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "Kumba",
        "type": "onsite",
        "category": "Marketing",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "cbe75d13-1ae1-49ae-a3cd-aa6f63716976",
        "title": "Software Engineering Internship - Summer 2025",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "San Francisco, CA",
        "type": "Full-time",
        "category": "Technology",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "85dfbf28-6e76-45a1-856c-34da3893c90c",
        "title": "email",
        "description": "No description available",
        "company": "Seed Company",
        "location": "fdfs",
        "type": "onsite",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "1914fd8c-85fe-418b-81e2-89fd752d5241",
        "title": "hghj",
        "description": "No description available",
        "company": "Seed Company",
        "location": "jkjjk",
        "type": "remote",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "9b9e1928-8f4a-4226-9c94-dfd5832ec079",
        "title": "jdh",
        "description": "No description available",
        "company": "Seed Company",
        "location": "sdsdds",
        "type": "remote",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "a3cd1643-8224-40ee-b17e-d2355ad67d17",
        "title": "Machine Laerning",
        "description": "No description available",
        "company": "Seed Company",
        "location": "Yde",
        "type": "onsite",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "0c2757d5-1f96-4aee-aa30-95f313cd3144",
        "title": "Frontend development",
        "description": "No description available",
        "company": "Digetech",
        "location": "Bambili",
        "type": "hybrid",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "0fe278c9-4873-4a9b-930f-3015fadf9481",
        "title": "Cyber Security",
        "description": "No description available",
        "company": "FutureProspect",
        "location": "Bamenda",
        "type": "onsite",
        "category": "Data Science",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "443da22c-377b-4831-821d-2e86abcc8930",
        "title": "Weekend Of Code",
        "description": "No description available",
        "company": "Seed Company",
        "location": "Bamenda",
        "type": "remote",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "336d819a-2c0c-425b-a9a0-1732ce552a17",
        "title": "jdhufusjfnj",
        "description": "No description available",
        "company": "SEEDINC",
        "location": "rtyuiop",
        "type": "onsite",
        "category": "Finance",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "bcbe2c41-9c07-421a-8e60-7d6e65650123",
        "title": "web Development",
        "description": "No description available",
        "company": "Fupro",
        "location": "Bambili",
        "type": "remote",
        "category": "Design",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "d4f263b7-dc64-41fe-bca7-0aa648678e9d",
        "title": "Networking Engineer",
        "description": "No description available",
        "company": "Fupro",
        "location": "Bafnaji",
        "type": "hybrid",
        "category": "Data Science",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "4ffc3ccf-5cdf-4624-b9e7-aceb1589c653",
        "title": "Cyber Security",
        "description": "No description available",
        "company": "Fupro",
        "location": "Bamenda",
        "type": "onsite",
        "category": "Software Development",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "1330a824-5b7b-4646-9352-ac278bc086c0",
        "title": "Internship Title",
        "description": "No description available",
        "company": "futureP",
        "location": "Location",
        "type": "onsite",
        "category": "Engineering",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "258e444c-651e-4140-aa51-35e26cc0e5a1",
        "title": "Frontend Come together",
        "description": "No description available",
        "company": "GIEFT",
        "location": "Remote",
        "type": "onsite",
        "category": "event",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "a0bac3d4-58ec-45e4-858e-a7946a169d19",
        "title": "Frontend Come together",
        "description": "No description available",
        "company": "GIEFT",
        "location": "Remote",
        "type": "onsite",
        "category": "event",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "7f2701aa-afa7-48e1-9dc3-e17209a07f87",
        "title": "Frontend Developer Intern",
        "description": "No description available",
        "company": "GIEFT",
        "location": "Remote",
        "type": "onsite",
        "category": "internship",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "e8961f92-dc23-4077-9efe-06f8de0616ac",
        "title": "Frontend Developer Intern",
        "description": "No description available",
        "company": "GIEFT",
        "location": "Remote",
        "type": "onsite",
        "category": "internship",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      },
      {
        "id": "62566ceb-6c0f-40a4-80fb-778c4c180c57",
        "title": "Cyber securtiy",
        "description": "No description available",
        "company": "GIEFT",
        "location": "Hybride",
        "type": "Type not specified",
        "category": "internship",
        "logo_url": "",
        "cover_image_url": "",
        "created_at": "2025-10-15T12:17:32.727Z"
      }
    ],
    "events": [
      {
        "id": "8866b5f4-1ea5-4349-b294-f8d02c289a12",
        "title": "WEEKEND OF CODE 2025 LAUNCH",
        "description": "Weekend of Code is a hands-on learning experience designed for absolute beginners and curious minds who want to explore the world of technology. Hosted by SEED, this program runs every Saturday and Sunday, bringing together passionate learners to discover, build, and grow through coding and innovation.\r\n\r\nOver the weekends, participants dive into practical sessions across five key tech domains:\r\n💡 Data Science\r\n🤖 Machine Learning\r\n🌐 Web Development\r\n🛡️ Cybersecurity\r\n📊 Project Management\r\n\r\nNo prior experience? No problem. Each session is crafted to help you start from zero — learning the fundamentals, trying out tools, and working on small but impactful projects that spark creativity and confidence.\r\n\r\nBy the end of each weekend, participants will not only gain real skills but also join a vibrant community of learners and mentors building the future of tech from Bamenda and beyond.",
        "company": "LearnCode",
        "start_date": "2025-11-06",
        "end_date": "2025-11-29",
        "location": "MILE 6- ADJACENT MAWA",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/LearnCode/events/WEEKEND%20OF%20CODE%202025%20LAUNCH-1760288810098.png",
        "created_at": "2025-10-12T17:06:51.314257+00:00"
      },
      {
        "id": "e1bc36e0-946a-4ed9-aab0-b1c326330d52",
        "title": "Web Talks",
        "description": "testing notification for the event",
        "company": "LearnCode",
        "start_date": "2025-11-01",
        "end_date": "2025-10-23",
        "location": "Onsite",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/LearnCode/events/Web%20Talks-1759396988910.png",
        "created_at": "2025-10-02T09:23:10.19135+00:00"
      },
      {
        "id": "1614587e-cf2b-4c60-bf42-01e8f8b886df",
        "title": "Ai talks",
        "description": "AI TATLS is a forward-looking event dedicated to exploring the transformative impact of Artificial Intelligence across industries, research, and society. Designed as a platform for knowledge exchange, innovation, and networking, the event brings together leading experts, researchers, entrepreneurs, and enthusiasts to discuss the latest trends, breakthroughs, and ethical considerations in AI.\r\n\r\nThrough keynote talks, panel discussions, and interactive sessions, participants will gain insights into:\r\n\r\nCutting-edge AI technologies shaping the future\r\n\r\nPractical applications of AI in business, healthcare, education, and beyond\r\n\r\nResponsible and ethical AI practices\r\n\r\nOpportunities for collaboration and innovation\r\n\r\nWhether you’re an AI professional, a student eager to learn, or an organization seeking to adopt AI solutions, AI TATLS offers valuable perspectives and connections to navigate the rapidly evolving AI landscape.",
        "company": "LearnCode",
        "start_date": "2025-10-01",
        "end_date": "2025-10-01",
        "location": "Online",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/LearnCode/events/Ai%20talks-1759348351273.png",
        "created_at": "2025-10-01T19:52:31.055012+00:00"
      },
      {
        "id": "3086dae6-e6d4-4e49-ac3f-ced738ce002b",
        "title": "Career Guidance Worshop",
        "description": "This is where purpose meets potentials All about knowing the giant inside of you",
        "company": "SEEDINC",
        "start_date": "2025-09-27",
        "end_date": "2025-09-28",
        "location": "Mile 6 nkwen",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/SEEDINC/events/Career%20Guidance%20Worshop-1758680292711.png",
        "created_at": "2025-09-24T02:18:13.994285+00:00"
      },
      {
        "id": "9e3ddca3-a032-44bb-9d33-aff64549b193",
        "title": "Trigger Test Event",
        "description": "Testing database trigger",
        "company": "FutureProspect",
        "start_date": "2025-07-01",
        "end_date": "2025-07-02",
        "location": "Virtual",
        "event_picture_url": "",
        "created_at": "2025-09-21T15:33:45.865238+00:00"
      },
      {
        "id": "d6709b2e-677c-4ca7-8dcf-212d27e10282",
        "title": "AI AWARENESS PROGRAM",
        "description": "AI Awareness Seminar – Cameroon\r\n\r\nArtificial Intelligence (AI) is transforming the way we live, work, and innovate across the globe. From smart assistants and healthcare solutions to education, agriculture, and business growth, AI is no longer just a future concept—it is today’s reality.\r\n\r\nThis AI Awareness Seminar in Cameroon is designed to introduce participants to the fundamentals of AI, its real-world applications, and the opportunities it creates for individuals, businesses, and communities. The event will break down complex concepts into simple, practical insights, helping attendees understand how AI can drive productivity, innovation, and sustainable development in Cameroon and Africa at large.",
        "company": "LearnCode",
        "start_date": "2025-09-03",
        "end_date": "2025-09-03",
        "location": "Physical",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/LearnCode/events/AI%20AWARENESS%20PROGRAM-1758190363116.png",
        "created_at": "2025-09-18T10:12:44.539056+00:00"
      },
      {
        "id": "fd2eb523-aca3-4a69-9766-132100bef0cd",
        "title": "AI Friday",
        "description": "SEED is an innovative technology company dedicated to delivering modern digital solutions that solve real-world challenges. We specialize in [your main services or products, e.g., web development, healthcare platforms, financial management systems], combining creativity, technology, and user-focused design to drive efficiency and impact. Our mission is to empower individuals and organizations through accessible, reliable, and scalable digital tools.",
        "company": "Seed Company",
        "start_date": "2025-09-12",
        "end_date": "2025-09-14",
        "location": "Bamenda",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/Seed%20Company/events/AI%20Friday-1757775680145.jpg",
        "created_at": "2025-09-13T15:01:30.247697+00:00"
      },
      {
        "id": "0ed96ec9-6157-4e90-bc62-31c6b3e139bd",
        "title": "SEED community Challenge",
        "description": "SEED is an innovative technology company dedicated to delivering modern digital solutions that solve real-world challenges. We specialize in [your main services or products, e.g., web development, healthcare platforms, financial management systems], combining creativity, technology, and user-focused design to drive efficiency and impact. Our mission is to empower individuals and organizations through accessible, reliable, and scalable digital tools.",
        "company": "Seed Company",
        "start_date": "2025-09-13",
        "end_date": "2025-09-14",
        "location": "remote",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/Seed%20Company/events/SEED%20community%20Challenge-1757769838240.jpg",
        "created_at": "2025-09-13T13:23:58.600154+00:00"
      },
      {
        "id": "c6643d2d-6724-4835-a3c5-c8c351af0aad",
        "title": "AI & Data Science Conference 2025",
        "description": "A conference bringing together researchers, engineers, and entrepreneurs to discuss the future of AI and Data Science.",
        "company": "Digetech",
        "start_date": "2025-11-15",
        "end_date": "2025-11-17",
        "location": "Yaoundé Conference Center, Cameroon",
        "event_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/company-assets/Digetech/events/AI%20&%20Data%20Science%20Conference%202025-1757427986843.png",
        "created_at": "2025-09-09T14:26:28.188599+00:00"
      }
    ],
    "programs": [
      {
        "id": "354d54b5-808b-4839-a6aa-dd2660d8730c",
        "title": "WEEKEND OF CODE 2025",
        "description": "Weekend of Code is a hands-on learning experience designed for absolute beginners and curious minds who want to explore the world of technology. Hosted by SEED, this program runs every Saturday and Sunday, bringing together passionate learners to discover, build, and grow through coding and innovation.\r\n\r\nOver the weekends, participants dive into practical sessions across five key tech domains:\r\n💡 Data Science\r\n🤖 Machine Learning\r\n🌐 Web Development\r\n🛡️ Cybersecurity\r\n📊 Project Management\r\n\r\nNo prior experience? No problem. Each session is crafted to help you start from zero — learning the fundamentals, trying out tools, and working on small but impactful projects that spark creativity and confidence.\r\n\r\nBy the end of each weekend, participants will not only gain real skills but also join a vibrant community of learners and mentors building the future of tech from Bamenda and beyond.",
        "organizer": "LearnCode",
        "program_category": "bootcamp",
        "start_date": "2025-10-12T00:00:00+00:00",
        "end_date": "2025-11-28T00:00:00+00:00",
        "program_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/program_pictures/d96eff03-b4a6-4110-8d75-51fe0569b3d9/cb5b97e7-ee32-4de3-803e-5227e35dd667.png",
        "created_at": "2025-10-12T16:58:21.863806+00:00"
      },
      {
        "id": "6ae4c458-25f2-44d6-acfb-1361c0e382bc",
        "title": "Program testoing notification",
        "description": "branding and publishing",
        "organizer": "LearnCode",
        "program_category": "volunteer",
        "start_date": "2025-10-23T00:00:00+00:00",
        "end_date": "2025-10-10T00:00:00+00:00",
        "program_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/program_pictures/d96eff03-b4a6-4110-8d75-51fe0569b3d9/de857e06-728d-411e-b7fe-55a60829bfa7.jpeg",
        "created_at": "2025-10-02T09:25:21.046723+00:00"
      },
      {
        "id": "9d130bac-d548-446a-b4f2-cf3bc4d69c73",
        "title": "Beyond the Browser at Seed HeadQuarter",
        "description": "Beyond the Browser is a SEED initiative designed to help participants move past traditional, surface-level web usage and explore the deeper layers of the digital world. This program equips learners with the knowledge and skills to understand, build, and innovate with technologies that power today’s internet and emerging digital ecosystems.\r\n\r\nThrough a mix of workshops, hands-on projects, and guided mentorship, participants will:\r\n\r\nExplore the fundamentals of how the web works “under the hood” — from networking and servers to APIs and cloud systems.\r\n\r\nGain practical experience in building applications and services that extend beyond web browsers, such as mobile apps, IoT integrations, and AI-driven tools.\r\n\r\nDevelop a critical understanding of cybersecurity, privacy, and the ethical use of technology.\r\n\r\nExperiment with cutting-edge innovations like decentralized systems, edge computing, and generative AI.",
        "organizer": "LearnCode",
        "program_category": "bootcamp",
        "start_date": "2025-09-30T00:00:00+00:00",
        "end_date": "2025-11-03T00:00:00+00:00",
        "program_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/program_pictures/d96eff03-b4a6-4110-8d75-51fe0569b3d9/87509071-22a9-4388-a73f-74565dab7183.png",
        "created_at": "2025-09-30T15:42:15.917815+00:00"
      },
      {
        "id": "9d961c1c-c24b-45b6-a88f-c538fa27cc6b",
        "title": "EPIC GATHERING",
        "description": "An epic tech gathering for all tech enthusiast. Networking with great minds, and sharing knowledge",
        "organizer": "SEEDINC",
        "program_category": "apprenticeship",
        "start_date": "2025-09-30T00:00:00+00:00",
        "end_date": "2025-09-30T00:00:00+00:00",
        "program_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/program_pictures/932f7544-e4fc-40d1-9908-d4cfc488a080/375fb84e-5b01-4929-8fae-d97f8ed24c59.png",
        "created_at": "2025-09-26T22:31:13.115895+00:00"
      },
      {
        "id": "15104c3d-28ca-4d70-ba9f-86ab99eb9ddb",
        "title": "AI & Machine Learning Bootcamp",
        "description": "Description & Activities:\r\n\r\nJoin our intensive 8-week remote Machine Learning Bootcamp designed to equip participants with hands-on experience in building and deploying real-world AI models. Whether you're a student, recent graduate, or career switcher, this program offers immersive training in key machine learning concepts.",
        "organizer": "Digetech",
        "program_category": "bootcamp",
        "start_date": "2025-09-22T00:00:00+00:00",
        "end_date": "2025-09-30T00:00:00+00:00",
        "program_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/program_pictures/a842c71b-1432-4e35-ba6c-89a83cea5637/cfeca397-5a4d-47d1-822b-e616cb6df41b.png",
        "created_at": "2025-09-22T05:51:07.344009+00:00"
      },
      {
        "id": "3deeaee8-a49d-4a3b-92e0-6d5cfb4be210",
        "title": "Trigger Test Program",
        "description": "Testing database trigger",
        "organizer": "FutureProspect",
        "program_category": "bootcamp",
        "start_date": "2025-06-01T00:00:00+00:00",
        "end_date": "2025-08-31T00:00:00+00:00",
        "program_picture_url": "",
        "created_at": "2025-09-21T15:33:45.865238+00:00"
      },
      {
        "id": "866823f8-8078-4cdf-952f-d6dd02bfa818",
        "title": "sds",
        "description": "asdfdafdfedfasdfdrf ffddav dsdf",
        "organizer": "Seed Company",
        "program_category": "bootcamp",
        "start_date": "2025-09-22T00:00:00+00:00",
        "end_date": "2025-09-15T00:00:00+00:00",
        "program_picture_url": "",
        "created_at": "2025-09-21T14:04:41.739629+00:00"
      },
      {
        "id": "6b27564d-c9de-4190-9fe7-7fd7882285fa",
        "title": "mentorship",
        "description": "jgh jhjh gfdf khj j drd",
        "organizer": "Seed Company",
        "program_category": "mentorship",
        "start_date": "2025-09-19T00:00:00+00:00",
        "end_date": "2025-09-22T00:00:00+00:00",
        "program_picture_url": "",
        "created_at": "2025-09-19T17:23:43.390891+00:00"
      },
      {
        "id": "6d2f07d1-9fd9-4ee2-88e6-8282c6656dc6",
        "title": "NextGen AI Hackathon 2024",
        "description": "A 3-day competitive event for developers and AI enthusiasts to build innovative solutions.",
        "organizer": "Seed Company",
        "program_category": "bootcamp",
        "start_date": "2025-09-07T00:00:00+00:00",
        "end_date": "2025-09-14T00:00:00+00:00",
        "program_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/program_pictures/a842c71b-1432-4e35-ba6c-89a83cea5637/f4b164dd-a126-4d46-8175-a592a7da69e5.jpg",
        "created_at": "2025-09-07T12:59:40.84281+00:00"
      },
      {
        "id": "86488d1e-c123-4b57-98a1-afebffaad678",
        "title": "cyber security program",
        "description": "we build something cool so be a part of this jurney and lets do this",
        "organizer": "futureP",
        "program_category": "bootcamp",
        "start_date": "2024-10-01T09:00:00+00:00",
        "end_date": "2024-12-31T17:00:00+00:00",
        "program_picture_url": "https://tmvipinvvhgklmqwvows.supabase.co/storage/v1/object/public/program_pictures/d34a80bb-5396-4bf5-a3f1-0a1642e423f7/55250e7a-72eb-46b0-a4fe-20f89061a0d9.jpg",
        "created_at": "2025-09-07T00:02:10.770499+00:00"
      }
    ]
  },
  "metadata": {
    "total_count": 51,
    "internships_count": 32,
    "events_count": 9,
    "programs_count": 10,
    "timestamp": "2025-10-15T12:17:32.730Z",
    "data_sources": {
      "internships": {
        "fetched": true,
        "count": 32
      },
      "events": {
        "fetched": true,
        "count": 9
      },
      "programs": {
        "fetched": true,
        "count": 10
      }
    }
  }
}"



