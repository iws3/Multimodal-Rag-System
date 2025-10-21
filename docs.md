Multimodal RAG System - Usage Guide

Overview

This repository provides a production-ready backend for a multimodal RAG (Retrieval-Augmented Generation) system. It exposes a small, high-level API that allows developers to:

- Generate embeddings from text
- Chunk raw text or arrays of objects
- Store documents and their chunks into PostgreSQL with pgvector vectors
- Retrieve semantically similar chunks by query
- Upsert documents idempotently

The design goal is to hide complexity from application developers so they can call simple endpoints from frontends or serverless functions.

Quick start

1. Environment

- Ensure you have a PostgreSQL instance with the pgvector extension installed/enabled.
- Set DATABASE_URL in your environment or .env file, e.g.:

postgresql+asyncpg://user:password@host:5432/dbname

2. Install Python deps

This project uses async SQLAlchemy and sentence-transformers. Install dependencies listed in `requirements.txt`:

python -m pip install -r requirements.txt

3. Run locally

Start the FastAPI app (example with uvicorn):

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Important API endpoints

Prefix: /api

- POST /api/embed
  - Body: { "texts": ["text1", "text2"] }
  - Response: { "embeddings": [[...], [...]] }

- POST /api/chunk
  - Body: { "text": "..." } or { "object_array": [ {...}, {...} ] }
  - Response: [ { "content": "..." }, ... ]

- POST /api/store
  - Body: {
      "document_type": "text",
      "title": "My Doc",
      "raw_text": "...",
      "metadata": { ... }
    }
  - Accepts pre-chunked `chunks`, `raw_text` string, or `objects` (array of JSON objects)
  - Response: { "document_id": "...", "chunks_stored": N }

- POST /api/retrieve
  - Body: { "query": "What is diabetes?", "top_k": 5 }
  - Response: { "query": "...", "results": [...], "total_results": N }

- POST /api/upsert
  - Similar to /api/store. If a document with the same title exists, replaces its chunks.

Notes and best practices

- Chunking: The default chunker is a simple whitespace-based splitter. It uses `TEXT_CHUNK_SIZE` and `TEXT_CHUNK_OVERLAP` from the app settings. For production, integrate a tokenizer-based chunker (e.g., tiktoken) for better token-level control.

- Embeddings: The repo uses `sentence-transformers` by default (model `all-MiniLM-L6-v2`). For higher-quality embeddings or latency/throughput tradeoffs, swap models in `app.config.setting` and ensure the model dimension matches `EMBEDDING_DIMENSION`.

- Database: Tables are created on startup by `init_db()` in `app.db.connection`. Ensure your DATABASE_URL points to a Postgres instance where your user can create extensions and tables. The app attempts to `CREATE EXTENSION IF NOT EXISTS vector;` on startup.

- Storing metadata: Documents and chunks store flexible JSON metadata via `meta_data` columns.

- Security: Protect your API endpoints when exposing them publicly (API keys, auth middleware, rate limiting). Disable SQLAlchemy echo in production.

Deploying to Render

1. Create a new Web Service in Render and point it to this repository.
2. Build command:

pip install -r requirements.txt

3. Start command:

uvicorn app.main:app --host 0.0.0.0 --port $PORT

4. Environment variables on Render:

- DATABASE_URL
- Any other secrets (e.g., model paths)

5. Postgres on Render: either provision a managed Postgres instance or use an external one. Ensure it has the `vector` extension.

Example client usage (curl)

Embed:

curl -X POST http://localhost:8000/api/embed -H "Content-Type: application/json" -d '{"texts": ["hello world"]}'

Store:

curl -X POST http://localhost:8000/api/store -H "Content-Type: application/json" -d '{"document_type":"text","title":"MyDoc","raw_text":"Hello world..."}'

Retrieve:

curl -X POST http://localhost:8000/api/retrieve -H "Content-Type: application/json" -d '{"query":"What is hello?","top_k":3}'

Upload (multimodal files)

You can upload files to `/api/upload` using multipart/form-data. The server currently performs lightweight processing:

- `.txt` and `.md` files: text is extracted and chunked.
- `.pdf`, images, audio, video: processing is stubbed and returns a note describing how to enable full processing.

Example (upload text file):

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@./example.txt"
```

Example (upload image):

```bash
curl -X POST "http://localhost:8000/api/upload" \
  -F "file=@./image.png"
```

The response will include a `note` field for stubbed types. To fully process PDFs/images/audio, integrate the respective extraction libraries and update `backend/app/processors/*`.

Troubleshooting

- Model memory: loading large embedding models consumes memory. Use smaller models for constrained hosts.
- Vector extension errors: Ensure the Postgres user has privileges to create extensions or pre-create the extension.
- Performance: For high throughput, batch embedding requests and use connection pooling.

Next steps / Enhancements

- Add authentication and rate-limiting middleware.
- Implement advanced chunkers (token-based), and file extractors for PDFs/DOCX.
- Add OpenAI/Gemini integration for richer LLM responses and reranking.
- Add unit tests for API endpoints and database interactions.

