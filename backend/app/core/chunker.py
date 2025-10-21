"""
Chunker module

Provides simple, dependency-light text chunking and functions to create
Document and Chunk records in the database. Designed to be conservative
and easy to use for beginners: accepts pre-extracted text for images,
audio, and video (so heavy external dependencies like OCR/Whisper are
optional and can be implemented in processors as needed).

Core functions:
 - chunk_text(text, chunk_size=500, overlap=50)
 - create_document_with_chunks(db, title, file_path, document_type, text, ...)

Notes:
 - Tokenization is approximated by splitting on whitespace. This keeps
   the implementation lightweight and avoids requiring a tokenizer.
 - Embeddings are generated via app.core.embeddings (batch when possible).
"""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import Document, Chunk
from app.core.embeddings import batch_generate_embeddings, generate_embedding
import uuid


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
	"""
	Splits a long text into overlapping chunks. 'chunk_size' and 'overlap'
	are measured in words (approximate tokens) to avoid adding heavy
	tokenizers as a hard dependency.

	Returns a list of chunk strings.
	"""
	if not text:
		return []

	words = text.split()
	if chunk_size <= 0:
		raise ValueError("chunk_size must be > 0")
	if overlap < 0:
		overlap = 0

	chunks: List[str] = []
	start = 0
	n = len(words)
	while start < n:
		end = min(start + chunk_size, n)
		chunk_words = words[start:end]
		chunks.append(" ".join(chunk_words))
		if end == n:
			break
		# Move start forward by chunk_size - overlap
		start += max(1, chunk_size - overlap)

	return chunks


async def create_document_with_chunks(
	db: AsyncSession,
	title: str,
	file_path: Optional[str],
	document_type: str,
	text: str,
	metadata: Optional[dict] = None,
	chunk_size: int = 500,
	overlap: int = 50,
	batch_embeddings: bool = True,
) -> Document:
	"""
	Create a Document record and associated Chunk records from pre-extracted
	text. This function generates embeddings for each chunk and stores them.

	- db: AsyncSession (SQLAlchemy)
	- text: pre-extracted text (for text files this can be read automatically
	  by processors; for images/audio/video this should be the transcription
	  or extracted text + visual description produced by processors)

	Returns the created Document instance (detached / not refreshed).
	"""
	if text is None:
		raise ValueError("text must be provided (for multimodal inputs provide extracted text)")

	# 1) Chunk the text
	chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

	# 2) Create Document
	doc = Document(
		id=uuid.uuid4(),
		document_type=document_type,
		title=title,
		file_path=file_path,
		meta_data=metadata or {},
	)

	await db.merge(doc)

	# 3) Generate embeddings (batch when helpful)
	embeddings = None
	if batch_embeddings and len(chunks) > 1:
		try:
			embeddings = batch_generate_embeddings(chunks)
		except Exception:
			embeddings = None

	# 4) Create chunk records
	for i, content in enumerate(chunks):
		emb = None
		if embeddings and i < len(embeddings):
			emb = embeddings[i]
		else:
			try:
				emb = generate_embedding(content)
			except Exception:
				emb = None

		chunk = Chunk(
			id=uuid.uuid4(),
			document_id=doc.id,
			content=content,
			chunk_index=i,
			embedding=emb,
			meta_data={"source": "auto_chunk", "chunk_words": len(content.split())},
		)
		await db.merge(chunk)

	await db.commit()
	return doc

