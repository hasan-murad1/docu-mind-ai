import sys
from pathlib import Path

# Allow importing from app/core
sys.path.append(str(Path(__file__).parent.parent / "app" / "core"))

from document_processor import chunk_text, extract_text


def test_chunk_text_basic_split():
    """A text with more words than chunk_size should be split into multiple chunks."""
    text = "word " * 50  # 50 words
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    assert len(chunks) > 1, "Text longer than chunk_size should produce multiple chunks"


def test_chunk_text_single_chunk_for_short_text():
    """A text shorter than chunk_size should produce exactly one chunk."""
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=100, overlap=10)

    assert len(chunks) == 1, "Short text should fit in a single chunk"


def test_chunk_text_overlap_preserves_words():
    """Consecutive chunks should share some overlapping words."""
    text = " ".join(f"word{i}" for i in range(30))  # word0, word1, ..., word29
    chunks = chunk_text(text, chunk_size=10, overlap=3)

    first_chunk_words = chunks[0].split()
    second_chunk_words = chunks[1].split()

    # Last 3 words of chunk 1 should appear at the start of chunk 2
    overlap_words = first_chunk_words[-3:]
    assert overlap_words == second_chunk_words[:3], "Overlap words should match between consecutive chunks"


def test_chunk_text_empty_string():
    """Empty text should produce no chunks (edge case, must not crash)."""
    chunks = chunk_text("", chunk_size=100, overlap=10)
    assert chunks == [], "Empty text should return an empty list"


def test_extract_text_unsupported_file_type():
    """Unsupported file extensions should raise a clear error, not crash silently."""
    try:
        extract_text("some_file.txt")
        assert False, "Should have raised ValueError for unsupported file type"
    except ValueError as e:
        assert "Unsupported file type" in str(e)


def test_extract_text_from_real_docx():
    """The sample test docx should extract non-empty text containing known content."""
    text = extract_text("data/sample_test.docx")
    assert len(text) > 0, "Extracted text should not be empty"
    assert "DocuMind AI" in text, "Extracted text should contain expected content"