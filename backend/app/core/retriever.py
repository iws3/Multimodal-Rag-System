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
