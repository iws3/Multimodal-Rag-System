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
    query_text = Column(Text, nullable=False)
    response = Column(Text, nullable=True)
    retrieved_chunk_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=True) # A list of UUIDs of chunks used for the response
    user_id = Column(String(255), nullable=True) # Optional: ID of the user who made the query
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<Query(id='{self.id}', text='{self.query_text[:50]}...')>"