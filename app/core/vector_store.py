import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Maximum chunks we allow in a single batch, to avoid memory issues
# with extremely large documents (safety limit).
MAX_CHUNKS_PER_BATCH = 5000

# Load the embedding model once (this is a bit slow, so we don't want
# to reload it every time we call a function).
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent ChromaDB client - saves data to disk in the "data" folder
# so embeddings survive between runs (not lost when the script ends).
_chroma_client = chromadb.PersistentClient(path="data/chroma_db")


def get_or_create_collection(collection_name: str = "documents"):
    """Get an existing ChromaDB collection, or create it if it doesn't exist."""
    return _chroma_client.get_or_create_collection(name=collection_name)


def add_chunks_to_store(
    chunks: list[str],
    source_filename: str,
    collection_name: str = "documents",
) -> int:
    """
    Embed a list of text chunks and store them in ChromaDB.
    Returns the number of chunks stored.
    """
    if not chunks:
        return 0

    if len(chunks) > MAX_CHUNKS_PER_BATCH:
        raise ValueError(
            f"Too many chunks ({len(chunks)}). Max allowed is {MAX_CHUNKS_PER_BATCH}."
        )

    collection = get_or_create_collection(collection_name)

    # Create embeddings for all chunks at once (batch processing is faster)
    embeddings = _embedding_model.encode(chunks).tolist()

    # ChromaDB needs a unique ID for each chunk
    ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]

    # Metadata helps us trace each chunk back to its source document
    metadatas = [{"source": source_filename, "chunk_index": i} for i in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def search_similar_chunks(
    query: str,
    top_k: int = 3,
    collection_name: str = "documents",
) -> list[dict]:
    """
    Search for the most semantically similar chunks to the query.
    Returns a list of dicts with 'text', 'source', and 'distance'.
    """
    collection = get_or_create_collection(collection_name)

    query_embedding = _embedding_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    output = []
    for i in range(len(results["documents"][0])):
        output.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })

    return output


# Manual test
if __name__ == "__main__":
    from document_processor import extract_text, chunk_text

    test_file = "data/sample_test.docx"
    text = extract_text(test_file)
    chunks = chunk_text(text, chunk_size=100, overlap=20)

    print(f"Storing {len(chunks)} chunks in ChromaDB...")
    count = add_chunks_to_store(chunks, source_filename=Path(test_file).name)
    print(f"Stored {count} chunks.")

    print("\n--- Searching for: 'What is this project about?' ---")
    results = search_similar_chunks("What is this project about?", top_k=2)
    for i, result in enumerate(results):
        print(f"\nResult {i+1} (distance: {result['distance']:.4f}):")
        print(result["text"])