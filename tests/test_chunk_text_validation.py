"""Tests for chunk_text input validation and windowing."""

import pytest

from document_processor import chunk_text


def make_words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (200, 200),   # overlap equal to chunk_size -> window never moves
        (200, 250),   # overlap bigger than chunk_size
        (0, 0),       # zero chunk size
        (-10, 0),     # negative chunk size
        (200, -1),    # negative overlap
        (200.0, 40),  # float instead of int
        ("200", 40),  # string instead of int
        (True, 0),    # bool is not accepted as an int
    ],
)
def test_chunk_text_rejects_bad_settings(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text here", chunk_size=chunk_size, overlap=overlap)


@pytest.mark.parametrize("bad_text", [None, 123, ["a", "b"]])
def test_chunk_text_rejects_non_string_text(bad_text):
    with pytest.raises(ValueError):
        chunk_text(bad_text, chunk_size=10, overlap=2)


def test_chunk_text_overlapping_windows():
    chunks = chunk_text(make_words(10), chunk_size=4, overlap=1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9", "w9"]


def test_chunk_text_zero_overlap():
    chunks = chunk_text(make_words(6), chunk_size=3, overlap=0)
    assert chunks == ["w0 w1 w2", "w3 w4 w5"]


@pytest.mark.parametrize("empty", ["", "   \n\t  "])
def test_chunk_text_empty_text_returns_empty_list(empty):
    assert chunk_text(empty, chunk_size=10, overlap=2) == []


def test_chunk_text_smallest_step_still_finishes():
    # overlap = chunk_size - 1 means the window moves 1 word at a time
    chunks = chunk_text(make_words(5), chunk_size=3, overlap=2)
    assert len(chunks) == 5
    assert chunks[-1] == "w4"


def test_chunk_text_app_settings():
    # Same settings as main.py: 200 words, 40 overlap -> step of 160
    chunks = chunk_text(make_words(500), chunk_size=200, overlap=40)
    assert len(chunks) == 4
    assert len(chunks[0].split()) == 200
    assert chunks[1].split()[0] == "w160"