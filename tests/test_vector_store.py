import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "app" / "core"))

from vector_store import add_chunks_to_store, search_similar_chunks

# Use a separate test collection so we never touch or pollute
# the real "documents" collection used by the actual app.
TEST_COLLECTION = "test_collection"


def test_add_chunks_returns_correct_count():
    """Storing chunks should return the number of chunks actually stored."""
    chunks = ["The sky is blue.", "Grass is green.", "Water is wet."]
    count = add_chunks_to_store(chunks, source_filename="test_doc.txt", collection_name=TEST_COLLECTION)

    assert count == 3, "Should report storing exactly 3 chunks"


def test_add_empty_chunks_returns_zero():
    """Storing an empty list of chunks should safely return 0, not crash."""
    count = add_chunks_to_store([], source_filename="empty.txt", collection_name=TEST_COLLECTION)
    assert count == 0, "Storing zero chunks should return 0"


def test_search_returns_relevant_result():
    """Searching for a topic should return the chunk that's actually about that topic."""
    chunks = [
        "Python is a popular programming language for data science.",
        "Bananas are a good source of potassium.",
        "The Great Wall of China is visible from certain satellite images.",
    ]
    add_chunks_to_store(chunks, source_filename="mixed_topics.txt", collection_name=TEST_COLLECTION)

    results = search_similar_chunks("What fruit is healthy?", top_k=1, collection_name=TEST_COLLECTION)

    assert len(results) == 1
    assert "Banana" in results[0]["text"], "Should retrieve the chunk about bananas for a fruit-related query"


def test_search_result_has_required_fields():
    """Each search result must include text, source, and distance."""
    chunks = ["This is a test sentence for field checking."]
    add_chunks_to_store(chunks, source_filename="field_test.txt", collection_name=TEST_COLLECTION)

    results = search_similar_chunks("test sentence", top_k=1, collection_name=TEST_COLLECTION)

    assert "text" in results[0]
    assert "source" in results[0]
    assert "distance" in results[0]