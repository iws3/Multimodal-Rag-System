from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.connection import get_db
from app.core.embeddings import generate_embedding, batch_generate_embeddings, load_embedding_model
from app.core.retriever import retrieve_similar_chunks
from app.config.setting import TEXT_CHUNK_SIZE, TEXT_CHUNK_OVERLAP
from app.db.models import Document, Chunk
import uuid
from fastapi import File, UploadFile
import shutil
import os
import mimetypes

router = APIRouter()

# ---------------------- Schemas ----------------------
class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

class ChunkItem(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class StoreRequest(BaseModel):
    document_type: str
    title: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunks: Optional[List[ChunkItem]] = None
    # alternative: accept raw text or array of objects
    raw_text: Optional[str] = None
    objects: Optional[List[Dict[str, Any]]] = None

class StoreResponse(BaseModel):
    document_id: str
    chunks_stored: int

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    document_type: Optional[str] = None
    min_similarity: float = 0.7

class RetrieveResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_results: int

# ---------------------- Helpers ----------------------

def simple_chunk_text(text: str, chunk_size: int = TEXT_CHUNK_SIZE, overlap: int = TEXT_CHUNK_OVERLAP) -> List[str]:
    """Naive chunker: splits on whitespace aiming for chunk_size characters with overlap."""
    if not text:
        return []
    words = text.split()
    chunks = []
    i = 0
    current = []
    current_len = 0
    while i < len(words):
        w = words[i]
        if current_len + len(w) + 1 <= chunk_size or not current:
            current.append(w)
            current_len += len(w) + 1
            i += 1
        else:
            chunks.append(" ".join(current))
            # create overlap
            overlap_words = current[-overlap:] if overlap < len(current) else current
            current = overlap_words.copy()
            current_len = sum(len(x) + 1 for x in current)
    if current:
        chunks.append(" ".join(current))
    return chunks

# ---------------------- Endpoints ----------------------
@router.post("/embed", response_model=EmbedResponse)
async def embed_text(req: EmbedRequest):
    """Generate embeddings for a list of texts."""
    try:
        # ensure model loaded
        await load_embedding_model()
    except Exception:
        pass
    embeddings = batch_generate_embeddings(req.texts)
    return EmbedResponse(embeddings=embeddings)

@router.post("/chunk", response_model=List[ChunkItem])
async def chunk_text_endpoint(text: Dict[str, Any]):
    """Accepts {"text": "..."} or {"object_array": [...] } and returns chunks."""
    if "text" in text and text["text"]:
        chunks = simple_chunk_text(text["text"])
        return [ChunkItem(content=c) for c in chunks]
    if "object_array" in text and isinstance(text["object_array"], list):
        # flatten objects -> json strings per object
        chunks = []
        for obj in text["object_array"]:
            import json
            s = json.dumps(obj)
            chunks.extend([ChunkItem(content=c) for c in simple_chunk_text(s)])
        return chunks
    # ---- New: accept arbitrary single-key payloads (e.g., {"additionalProp1": {...}}) ----
    # If caller sent a single key with a nested object, stringify or handle lists.
    if isinstance(text, dict) and len(text) == 1:
        key, val = next(iter(text.items()))
        # If value is a string, chunk it
        if isinstance(val, str) and val:
            chunks = simple_chunk_text(val)
            return [ChunkItem(content=c) for c in chunks]

        # If value is a dict, stringify and chunk
        if isinstance(val, dict):
            import json
            s = json.dumps(val)
            chunks = simple_chunk_text(s)
            return [ChunkItem(content=c) for c in chunks]

        # If value is a list, treat as object_array if list of dicts, otherwise stringify elements
        if isinstance(val, list):
            # list of dicts -> treat as object_array
            if all(isinstance(x, dict) for x in val):
                chunks = []
                import json
                for obj in val:
                    s = json.dumps(obj)
                    chunks.extend([ChunkItem(content=c) for c in simple_chunk_text(s)])
                return chunks
            # list of strings -> chunk each string
            if all(isinstance(x, str) for x in val):
                chunks = []
                for s in val:
                    chunks.extend([ChunkItem(content=c) for c in simple_chunk_text(s)])
                return chunks

    # Fallback: stringify the entire payload and chunk
    try:
        import json
        s = json.dumps(text)
        chunks = simple_chunk_text(s)
        return [ChunkItem(content=c) for c in chunks]
    except Exception:
        raise HTTPException(status_code=400, detail="Request must include 'text' or 'object_array' or be a single nested object/list.")

@router.post("/store", response_model=StoreResponse)
async def store_document(req: StoreRequest, db: AsyncSession = Depends(get_db)):
    """Store a document and its chunks into the database. Accepts pre-chunked data or raw text/objects."""
    doc_id = uuid.uuid4()
    doc = Document(
        id=doc_id,
        document_type=req.document_type,
        title=req.title,
        file_path=req.file_path,
        meta_data=req.metadata
    )
    await db.merge(doc)

    chunks_to_store = []
    if req.chunks:
        for i, ch in enumerate(req.chunks):
            chunks_to_store.append((i, ch.content, ch.metadata))
    elif req.raw_text:
        text_chunks = simple_chunk_text(req.raw_text)
        for i, c in enumerate(text_chunks):
            chunks_to_store.append((i, c, None))
    elif req.objects:
        import json
        obj_texts = [json.dumps(o) for o in req.objects]
        for idx, ot in enumerate(obj_texts):
            for j, c in enumerate(simple_chunk_text(ot)):
                chunks_to_store.append((idx * 1000 + j, c, None))
    else:
        raise HTTPException(status_code=400, detail="No chunk/source provided (provide 'chunks' or 'raw_text' or 'objects').")

    # Generate embeddings in batches
    texts = [c[1] for c in chunks_to_store]
    try:
        await load_embedding_model()
    except Exception:
        pass
    embeddings = batch_generate_embeddings(texts)

    # Persist chunks
    stored_count = 0
    for (idx, content, metadata), emb in zip(chunks_to_store, embeddings):
        chunk = Chunk(
            id=uuid.uuid4(),
            document_id=doc_id,
            content=content,
            chunk_index=idx,
            embedding=emb,
            meta_data=metadata
        )
        await db.merge(chunk)
        stored_count += 1
    await db.commit()

    return StoreResponse(document_id=str(doc_id), chunks_stored=stored_count)

@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest, db: AsyncSession = Depends(get_db)):
    try:
        results = await retrieve_similar_chunks(
            db=db,
            query_text=req.query,
            top_k=req.top_k,
            document_type=req.document_type,
            min_similarity=req.min_similarity
        )
        return RetrieveResponse(query=req.query, results=results, total_results=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upsert", response_model=StoreResponse)
async def upsert_document(req: StoreRequest, db: AsyncSession = Depends(get_db)):
    """Convenience endpoint to store or update a document (idempotent by document title + file_path)."""
    # Lookup by title+file_path to decide update vs create
    from sqlalchemy import select
    stmt = select(Document).where(Document.title == req.title)
    existing = await db.execute(stmt)
    row = existing.scalars().first()
    if row:
        # Delete existing chunks for that document
        await db.execute("DELETE FROM chunks WHERE document_id = :doc_id", {"doc_id": str(row.id)})
        await db.commit()
        # Treat as new store but reuse document id
        req_doc = StoreRequest(**req.dict())
        req_doc_id = row.id
        # create chunks similarly
        chunks_to_store = []
        if req.chunks:
            for i, ch in enumerate(req.chunks):
                chunks_to_store.append((i, ch.content, ch.metadata))
        elif req.raw_text:
            text_chunks = simple_chunk_text(req.raw_text)
            for i, c in enumerate(text_chunks):
                chunks_to_store.append((i, c, None))
        elif req.objects:
            import json
            obj_texts = [json.dumps(o) for o in req.objects]
            for idx, ot in enumerate(obj_texts):
                for j, c in enumerate(simple_chunk_text(ot)):
                    chunks_to_store.append((idx * 1000 + j, c, None))
        else:
            raise HTTPException(status_code=400, detail="No chunk/source provided.")

        texts = [c[1] for c in chunks_to_store]
        try:
            await load_embedding_model()
        except Exception:
            pass
        embeddings = batch_generate_embeddings(texts)

        stored_count = 0
        for (idx, content, metadata), emb in zip(chunks_to_store, embeddings):
            chunk = Chunk(
                id=uuid.uuid4(),
                document_id=req_doc_id,
                content=content,
                chunk_index=idx,
                embedding=emb,
                meta_data=metadata
            )
            await db.merge(chunk)
            stored_count += 1
        await db.commit()
        return StoreResponse(document_id=str(req_doc_id), chunks_stored=stored_count)
    else:
        # Create new
        return await store_document(req, db)


# ---------------------- Upload endpoint (multimodal) ----------------------
@router.post("/upload")
async def upload_file(file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    """Accept a single file upload and perform lightweight processing.

    - For .txt/.md: extract text and return chunks
    - For images/audio/video/pdf: return a processing stub response (simulating OCR/transcription)
    The endpoint does not yet persist unless you call `/api/store` with the extracted text.
    """
    filename = file.filename
    content_type = file.content_type or mimetypes.guess_type(filename)[0]

    # Save to a temporary location for inspection
    tmp_dir = os.path.join(os.getcwd(), "tmp_uploads")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Simple handlers
    lower = filename.lower()
    if lower.endswith(".txt") or lower.endswith(".md"):
        # read text and chunk
        with open(tmp_path, "r", encoding="utf-8", errors="ignore") as rf:
            text = rf.read()
        chunks = simple_chunk_text(text)
        return {"filename": filename, "type": "text", "chunks": chunks}

    if lower.endswith(".pdf"):
        # PDF processing stub
        return {"filename": filename, "type": "pdf", "note": "PDF processing stubbed. Integrate pdfminer or pypdf to extract text."}

    if any(lower.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
        # Image processing stub
        return {"filename": filename, "type": "image", "note": "Image OCR stubbed. Integrate Tesseract or Vision API for OCR."}

    if any(lower.endswith(ext) for ext in [".mp3", ".wav", ".m4a"]):
        # Audio processing stub
        return {"filename": filename, "type": "audio", "note": "Audio transcription stubbed. Integrate Whisper or other STT model."}

    if any(lower.endswith(ext) for ext in [".mp4", ".mov", ".avi"]):
        # Video processing stub
        return {"filename": filename, "type": "video", "note": "Video processing stubbed. Integrate ffmpeg + transcription + vision."}

    return {"filename": filename, "type": content_type or "unknown", "note": "No specialized processing available for this file type."}


# ---------------------- Diagnostic endpoints ----------------------
@router.get("/doc_types")
async def list_document_types(db: AsyncSession = Depends(get_db)):
    """Return distinct document_type values present in the documents table.

    Useful for debugging queries that filter by document_type.
    """
    try:
        result = await db.execute("SELECT DISTINCT document_type FROM documents;")
        types = [row[0] for row in result.fetchall() if row[0] is not None]
        return {"document_types": types, "count": len(types)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample_documents")
async def sample_documents(limit: int = 10, db: AsyncSession = Depends(get_db)):
    """Return a small sample of documents (id, title, type, snippet) for inspection."""
    try:
        result = await db.execute("SELECT id, title, document_type, meta_data FROM documents ORDER BY created_at DESC LIMIT :limit", {"limit": limit})
        rows = result.fetchall()
        items = []
        for r in rows:
            items.append({
                "id": str(r.id),
                "title": r.title,
                "document_type": r.document_type,
                "meta_data": r.meta_data
            })
        return {"documents": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DebugSimRequest(BaseModel):
    query: str
    top_k: int = 5
    document_type: Optional[str] = None


@router.post("/debug_similarity")
async def debug_similarity(req: DebugSimRequest, db: AsyncSession = Depends(get_db)):
    """Return the raw distance and similarity values for the top_k chunks for the query.

    This bypasses min_similarity thresholding and returns the numeric values helpful for tuning.
    """
    from app.core.retriever import retrieve_similar_chunks
    # Use retriever to get candidates but with min_similarity=0 to capture all
    results = await retrieve_similar_chunks(db=db, query_text=req.query, top_k=req.top_k, document_type=req.document_type, min_similarity=0.0)
    # The retriever now includes 'distance' in each item (when available)
    return {"query": req.query, "results": results, "total_results": len(results)}
