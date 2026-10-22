"""Tests for vector_store: storage (upsert) and the three search tools."""

import uuid

import pytest

import vector_store
from vector_store import (
    MAX_CHUNKS_PER_BATCH,
    add_chunks_to_store,
    keyword_search,
    list_sources,
    metadata_search,
    search_similar_chunks,
)


@pytest.fixture
def collection_name():
    """A fresh, unique collection per test, deleted afterwards."""
    name = f"test_{uuid.uuid4().hex[:12]}"
    yield name
    try:
        vector_store._chroma_client.delete_collection(name=name)
    except Exception:
        pass  # collection may never have been created


def count_chunks(collection_name):
    return vector_store.get_or_create_collection(collection_name).count()


# ---------- Storage ----------

def test_add_chunks_returns_correct_count(collection_name):
    chunks = ["The sky is blue.", "Grass is green.", "Water is wet."]
    count = add_chunks_to_store(chunks, "test_doc.txt", collection_name=collection_name)
    assert count == 3


def test_add_empty_chunks_returns_zero(collection_name):
    assert add_chunks_to_store([], "empty.txt", collection_name=collection_name) == 0


def test_add_too_many_chunks_raises(collection_name):
    with pytest.raises(ValueError):
        add_chunks_to_store(
            ["x"] * (MAX_CHUNKS_PER_BATCH + 1), "big.txt", collection_name=collection_name
        )


@pytest.mark.parametrize(
    "chunks, source",
    [
        ("not a list", "f.txt"),
        (["ok", 5], "f.txt"),
        (["ok"], ""),
        (["ok"], None),
    ],
)
def test_add_rejects_bad_input(collection_name, chunks, source):
    with pytest.raises(ValueError):
        add_chunks_to_store(chunks, source, collection_name=collection_name)


def test_reupload_same_file_does_not_duplicate(collection_name):
    chunks = ["one", "two", "three"]
    add_chunks_to_store(chunks, "f.txt", collection_name=collection_name)
    add_chunks_to_store(chunks, "f.txt", collection_name=collection_name)
    assert count_chunks(collection_name) == 3


def test_reupload_with_fewer_chunks_removes_old_ones(collection_name):
    add_chunks_to_store(["one", "two", "three"], "f.txt", collection_name=collection_name)
    add_chunks_to_store(["only one now"], "f.txt", collection_name=collection_name)
    assert count_chunks(collection_name) == 1
    results = search_similar_chunks("one", top_k=5, collection_name=collection_name)
    assert [r["text"] for r in results] == ["only one now"]


# ---------- Vector search ----------

def test_search_returns_relevant_result(collection_name):
    chunks = [
        "Python is a popular programming language for data science.",
        "Bananas are a good source of potassium.",
        "The Great Wall of China is visible from certain satellite images.",
    ]
    add_chunks_to_store(chunks, "mixed_topics.txt", collection_name=collection_name)
    results = search_similar_chunks("What fruit is healthy?", top_k=1, collection_name=collection_name)
    assert len(results) == 1
    assert "Banana" in results[0]["text"]


def test_search_result_has_required_fields(collection_name):
    add_chunks_to_store(["This is a test sentence."], "field_test.txt", collection_name=collection_name)
    results = search_similar_chunks("test sentence", top_k=1, collection_name=collection_name)
    assert {"text", "source", "distance"} <= set(results[0].keys())


def test_search_empty_collection_returns_empty_list(collection_name):
    assert search_similar_chunks("anything", top_k=3, collection_name=collection_name) == []


def all_tools(collection_name):
    """The three search tools with the same (query, top_k) call shape."""
    return [
        lambda q, k: search_similar_chunks(q, top_k=k, collection_name=collection_name),
        lambda q, k: keyword_search(q, top_k=k, collection_name=collection_name),
        lambda q, k: metadata_search(q, "a.txt", top_k=k, collection_name=collection_name),
    ]


@pytest.mark.parametrize("bad_query", ["", "   ", None, "x" * 501])
def test_all_tools_reject_bad_query(collection_name, bad_query):
    for tool in all_tools(collection_name):
        with pytest.raises(ValueError):
            tool(bad_query, 3)


@pytest.mark.parametrize("bad_top_k", [0, 21, "3", True, 2.5])
def test_all_tools_reject_bad_top_k(collection_name, bad_top_k):
    for tool in all_tools(collection_name):
        with pytest.raises(ValueError):
            tool("valid question", bad_top_k)


# ---------- Keyword search ----------

KEYWORD_CHUNKS = [
    "Invoice INV-2024-0042 was paid in full.",
    "The weather is sunny today.",
    "Our team uses Python for data work.",
]


def test_keyword_search_finds_exact_term(collection_name):
    add_chunks_to_store(KEYWORD_CHUNKS, "k.txt", collection_name=collection_name)
    results = keyword_search("Status of INV-2024-0042?", top_k=3, collection_name=collection_name)
    assert len(results) == 1
    assert "INV-2024-0042" in results[0]["text"]
    assert results[0]["score"] == 0.75  # 3 of 4 terms: inv, 2024, 0042 (not "status")
    assert {"text", "source", "distance", "score"} <= set(results[0].keys())
    assert isinstance(results[0]["distance"], float)


def test_keyword_search_is_case_insensitive(collection_name):
    add_chunks_to_store(KEYWORD_CHUNKS, "k.txt", collection_name=collection_name)
    results = keyword_search("PYTHON", top_k=3, collection_name=collection_name)
    assert "Python" in results[0]["text"]


def test_keyword_search_no_match_returns_empty(collection_name):
    add_chunks_to_store(KEYWORD_CHUNKS, "k.txt", collection_name=collection_name)
    assert keyword_search("zebra quantum", top_k=3, collection_name=collection_name) == []


def test_keyword_search_only_stopwords_returns_empty(collection_name):
    add_chunks_to_store(KEYWORD_CHUNKS, "k.txt", collection_name=collection_name)
    assert keyword_search("what is the", top_k=3, collection_name=collection_name) == []


def test_keyword_distance_matches_vector_distance(collection_name):
    add_chunks_to_store(["Bananas are a good source of potassium."], "b.txt", collection_name=collection_name)
    kw = keyword_search("bananas potassium", top_k=1, collection_name=collection_name)
    vec = search_similar_chunks("bananas potassium", top_k=1, collection_name=collection_name)
    assert kw[0]["distance"] == pytest.approx(vec[0]["distance"], abs=1e-3)


# ---------- Metadata search ----------

def test_metadata_search_only_returns_that_file(collection_name):
    add_chunks_to_store(["Apples are red.", "Apples are sweet."], "a.txt", collection_name=collection_name)
    add_chunks_to_store(["Apples grow on trees."], "b.txt", collection_name=collection_name)
    results = metadata_search("apples", "a.txt", top_k=5, collection_name=collection_name)
    assert len(results) == 2
    assert all(r["source"] == "a.txt" for r in results)


def test_metadata_search_unknown_file_returns_empty(collection_name):
    add_chunks_to_store(["Apples are red."], "a.txt", collection_name=collection_name)
    assert metadata_search("apples", "nope.txt", top_k=3, collection_name=collection_name) == []


def test_list_sources_is_sorted_and_unique(collection_name):
    add_chunks_to_store(["x one", "x two"], "b.txt", collection_name=collection_name)
    add_chunks_to_store(["y one"], "a.txt", collection_name=collection_name)
    assert list_sources(collection_name) == ["a.txt", "b.txt"]