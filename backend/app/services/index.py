"""
Beginner-friendly indexing services.

Expose a couple of simple async functions that beginners can call from
API endpoints or scripts to index text or image content into the DB.

These functions are intentionally small and wrap the lower-level
chunker and processor utilities.
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.chunker import create_document_with_chunks
from app.processors.text_processor import clean_text
from app.processors.image_processor import build_image_text_representation


async def index_plain_text(
    db: AsyncSession,
    title: str,
    file_path: Optional[str],
    text: str,
    metadata: Optional[dict] = None,
    chunk_size: int = 500,
    overlap: int = 50,
):
    """Index a plain text document. Returns the created Document.

    This is the function beginners should call for text uploads.
    """
    cleaned = clean_text(text)
    return await create_document_with_chunks(
        db=db,
        title=title,
        file_path=file_path,
        document_type="text",
        text=cleaned,
        metadata=metadata,
        chunk_size=chunk_size,
        overlap=overlap,
    )


async def index_image(
    db: AsyncSession,
    title: str,
    file_path: Optional[str],
    ocr_text: Optional[str],
    description: Optional[str],
    metadata: Optional[dict] = None,
    chunk_size: int = 500,
    overlap: int = 50,
):
    """Index an image by combining OCR text + description and creating chunks.

    The caller is responsible for performing OCR or vision analysis. This
    function simply normalizes and indexes the textual representation.
    """
    combined = build_image_text_representation(ocr_text, description)
    cleaned = clean_text(combined)
    return await create_document_with_chunks(
        db=db,
        title=title,
        file_path=file_path,
        document_type="image",
        text=cleaned,
        metadata=metadata,
        chunk_size=chunk_size,
        overlap=overlap,
    )
