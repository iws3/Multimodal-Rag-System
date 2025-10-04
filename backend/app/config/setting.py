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