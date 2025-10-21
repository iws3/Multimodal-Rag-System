from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends, HTTPException
from fastapi import status
from pydantic import BaseModel
from typing import Optional, Any
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.connection import get_db
from app.services.index import index_plain_text, index_image
from app.processors.text_processor import read_plain_text

router = APIRouter()


class UploadJSON(BaseModel):
	title: str
	document_type: str = "text"  # text | image | audio | video | json
	text: Optional[str] = None
	ocr_text: Optional[str] = None
	description: Optional[str] = None
	metadata: Optional[dict] = None


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_file(
	background_tasks: BackgroundTasks,
	file: Optional[UploadFile] = File(None),
	payload: Optional[UploadJSON] = None,
	db: AsyncSession = Depends(get_db),
):
	"""Accepts a file (multipart) or a JSON payload describing content.

	The actual processing is run in a background task so the endpoint
	returns quickly while the heavy lifting happens asynchronously.
	"""
	if not file and not payload:
		raise HTTPException(status_code=400, detail="Either file or payload must be provided")

	# If a file is provided, try to decode if it's plain text. For images/audio/video
	# we expect the client to include `payload` fields with OCR/transcript/description
	# or to call a separate service that performs OCR/transcription before calling this endpoint.
	if file:
		filename = file.filename or "uploaded"
		content_type = file.content_type or "application/octet-stream"

		# Quick handling for plain text files
		if content_type.startswith("text/") or filename.lower().endswith(('.txt', '.md', '.json')):
			body = await file.read()
			text = read_plain_text(body)

			# Schedule background indexing
			background_tasks.add_task(
				lambda: None
			)

			# Use background function that is coroutine-aware
			async def task_text_index():
				await index_plain_text(db=db, title=filename, file_path=None, text=text, metadata=None)

			background_tasks.add_task(lambda: __import__("asyncio").get_event_loop().create_task(task_text_index()))

			return {"status": "accepted", "message": "Text file accepted and will be indexed in background."}

		# For images/audio/video we accept the upload but require payload to contain OCR/transcript
		# If payload isn't provided, return an accepted status but request the client call another endpoint
		return {"status": "accepted", "message": "File uploaded. For non-text files, include OCR/transcript in payload to index automatically."}

	# If no file but JSON payload provided
	if payload:
		if payload.document_type == "text" and payload.text:
			async def t():
				await index_plain_text(db=db, title=payload.title, file_path=None, text=payload.text, metadata=payload.metadata)

			background_tasks.add_task(lambda: __import__("asyncio").get_event_loop().create_task(t()))
			return {"status": "accepted", "message": "Text payload accepted and will be indexed in background."}

		if payload.document_type == "image":
			# require at least ocr_text or description
			if not payload.ocr_text and not payload.description:
				raise HTTPException(status_code=400, detail="For images, provide ocr_text or description in payload")

			async def t2():
				await index_image(db=db, title=payload.title, file_path=None, ocr_text=payload.ocr_text, description=payload.description, metadata=payload.metadata)

			background_tasks.add_task(lambda: __import__("asyncio").get_event_loop().create_task(t2()))
			return {"status": "accepted", "message": "Image payload accepted and will be indexed in background."}

		# For audio/video/json just accept and instruct to include transcription
		return {"status": "accepted", "message": "Payload accepted. For audio/video provide transcription; for JSON provide textual field to index."}

