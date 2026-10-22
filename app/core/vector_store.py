"""
Vector store helpers: storing chunks and three read-only search tools.

Search tools (used by the agent, all read-only):
- search_similar_chunks: semantic (vector) search
- keyword_search: exact keyword overlap search (case-insensitive)
- metadata_search: vector search limited to one source file

All search tools return dicts with the same keys ('text', 'source',
'distance'), so the agent can apply one relevance filter to any tool.
"""

import re
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Safety limits ---
MAX_CHUNKS_PER_BATCH = 5000        # max chunks stored in one call
MAX_TOP_K = 20                     # max results any search can return
MAX_QUERY_LENGTH = 500             # characters
MAX_KEYWORD_SCAN_CHUNKS = 20000    # keyword search never scans more than this

# Common words ignored by keyword search (they match almost everything)
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in",
    "on", "at", "to", "for", "and", "or", "not", "what", "which", "who",
    "how", "when", "where", "why", "does", "do", "did", "this", "that",
    "these", "those", "it", "its", "with", "by", "from", "about", "as",
    "can", "i", "me", "my", "you", "your", "there", "any", "tell",
}

# Load the embedding model once (slow, so we don't reload it per call)
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent ChromaDB client - data survives between runs
_chroma_client = chromadb.PersistentClient(path="data/chroma_db")


# ---------- Input validation ----------

def _validate_query(query) -> str:
    """Return a clean query string, or raise ValueError."""
    if not isinstance(query, str):
        raise ValueError("Query must be a string.")
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query too long. Max {MAX_QUERY_LENGTH} characters.")
    return query


def _validate_top_k(top_k) -> int:
    """Return top_k if it is an int in [1, MAX_TOP_K], else raise ValueError."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise ValueError("top_k must be an integer.")
    if not 1 <= top_k <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}.")
    return top_k


def _validate_source_filename(source_filename) -> str:
    """Return a clean source filename, or raise ValueError."""
    if not isinstance(source_filename, str) or not source_filename.strip():
        raise ValueError("source_filename must be a non-empty string.")
    return source_filename.strip()


# ---------- Storage ----------

def get_or_create_collection(collection_name: str = "documents"):
    """Get an existing ChromaDB collection, or create it if it doesn't exist."""
    return _chroma_client.get_or_create_collection(name=collection_name)


def add_chunks_to_store(
    chunks: list[str],
    source_filename: str,
    collection_name: str = "documents",
) -> int:
    """
    Embed text chunks and store them in ChromaDB.

    Re-uploading the same file replaces its old chunks (no duplicates,
    no stale leftovers). Returns the number of chunks stored.
    """
    source_filename = _validate_source_filename(source_filename)
    if not isinstance(chunks, list) or not all(isinstance(c, str) for c in chunks):
        raise ValueError("chunks must be a list of strings.")
    if not chunks:
        return 0
    if len(chunks) > MAX_CHUNKS_PER_BATCH:
        raise ValueError(
            f"Too many chunks ({len(chunks)}). Max allowed is {MAX_CHUNKS_PER_BATCH}."
        )

    collection = get_or_create_collection(collection_name)

    # Embed first: if this fails, the old chunks are still untouched
    embeddings = _embedding_model.encode(chunks).tolist()
    ids = [f"{source_filename}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": source_filename, "chunk_index": i} for i in range(len(chunks))]

    # Remove the previous version of this file, then write the new one
    collection.delete(where={"source": source_filename})
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )
    return len(chunks)


def list_sources(collection_name: str = "documents") -> list[str]:
    """Return the sorted, unique source filenames stored in the collection."""
    collection = get_or_create_collection(collection_name)
    data = collection.get(limit=MAX_KEYWORD_SCAN_CHUNKS, include=["metadatas"])
    return sorted({m["source"] for m in data["metadatas"] if m and m.get("source")})


# ---------- Read-only search tools ----------

def _format_query_results(results) -> list[dict]:
    """Turn a raw ChromaDB query result into a list of simple dicts."""
    if not results["documents"]:
        return []
    output = []
    for i, text in enumerate(results["documents"][0]):
        output.append({
            "text": text,
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i],
        })
    return output


def search_similar_chunks(
    query: str,
    top_k: int = 3,
    collection_name: str = "documents",
) -> list[dict]:
    """
    Semantic search: return the chunks closest in meaning to the query.
    Each result has 'text', 'source', and 'distance' (lower = closer).
    """
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)

    collection = get_or_create_collection(collection_name)
    total = collection.count()
    if total == 0:
        return []

    query_embedding = _embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, total),
    )
    return _format_query_results(results)


def metadata_search(
    query: str,
    source_filename: str,
    top_k: int = 3,
    collection_name: str = "documents",
) -> list[dict]:
    """
    Semantic search limited to chunks from one source file.
    Returns [] if that file is not in the collection.
    """
    query = _validate_query(query)
    source_filename = _validate_source_filename(source_filename)
    top_k = _validate_top_k(top_k)

    collection = get_or_create_collection(collection_name)
    where = {"source": source_filename}

    # Count matching chunks first so we never ask for more than exist
    matching = len(collection.get(where=where, include=["metadatas"])["ids"])
    if matching == 0:
        return []

    query_embedding = _embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, matching),
        where=where,
    )
    return _format_query_results(results)


def _tokenize(text: str) -> list[str]:
    """Lowercase words/numbers, without stopwords."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in STOPWORDS]


def _distances_for_ids(query: str, ids: list[str], collection) -> dict:
    """
    Compute the same distance Chroma uses (squared L2) between the query
    and the given chunk IDs, so keyword results can use the same threshold.
    """
    query_vec = np.asarray(_embedding_model.encode([query])[0], dtype=np.float64)
    stored = collection.get(ids=ids, include=["embeddings"])
    distances = {}
    for chunk_id, embedding in zip(stored["ids"], stored["embeddings"]):
        diff = query_vec - np.asarray(embedding, dtype=np.float64)
        distances[chunk_id] = float(np.dot(diff, diff))
    return distances


def keyword_search(
    query: str,
    top_k: int = 3,
    collection_name: str = "documents",
) -> list[dict]:
    """
    Keyword search: rank chunks by how many query words they contain
    (case-insensitive, stopwords ignored). Good for exact names, numbers, IDs.
    Each result has 'text', 'source', 'distance', and 'score' (0-1 keyword match).
    """
    query = _validate_query(query)
    top_k = _validate_top_k(top_k)

    terms = set(_tokenize(query))
    if not terms:
        return []

    collection = get_or_create_collection(collection_name)
    data = collection.get(
        limit=MAX_KEYWORD_SCAN_CHUNKS,
        include=["documents", "metadatas"],
    )

    scored = []
    for chunk_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        matched = len(terms & set(_tokenize(text)))
        if matched:
            scored.append((matched / len(terms), chunk_id, text, meta["source"]))

    if not scored:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    top = scored[:top_k]
    distances = _distances_for_ids(query, [item[1] for item in top], collection)

    return [
        {
            "text": text,
            "source": source,
            "distance": distances[chunk_id],
            "score": round(score, 4),
        }
        for score, chunk_id, text, source in top
    ]


# Manual test: run from the project root
if __name__ == "__main__":
    from document_processor import extract_text, chunk_text

    test_file = "data/sample_test.docx"
    source = Path(test_file).name
    chunks = chunk_text(extract_text(test_file), chunk_size=200, overlap=40)

    # Store twice on purpose: the second upload should replace, not duplicate
    for attempt in range(2):
        count = add_chunks_to_store(chunks, source_filename=source)
        print(f"Upload {attempt + 1}: stored {count} chunks")

    stored = get_or_create_collection().get(where={"source": source}, include=["metadatas"])
    print(f"Chunks in DB for {source}: {len(stored['ids'])}")
    print("Sources:", list_sources())

    query = "What is this project about?"
    for name, results in [
        ("vector", search_similar_chunks(query, top_k=2)),
        ("keyword", keyword_search(query, top_k=2)),
        ("metadata", metadata_search(query, source, top_k=2)),
    ]:
        print(f"\n--- {name} search ---")
        for r in results:
            print(f"{r['source']} | distance={r['distance']:.4f} | {r['text'][:80]}")