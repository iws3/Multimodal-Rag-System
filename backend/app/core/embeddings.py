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


def load_embedding_model():
    """
    Loads the SentenceTransformer model. This should be called once at application startup.

    This function is synchronous because many parts of the code (and FastAPI
    startup) call it synchronously. Keeping it synchronous avoids accidental
    coroutine misuse where code calls it without awaiting.
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
        load_embedding_model()
        if embedding_model is None:
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