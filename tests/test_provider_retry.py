"""Tests for provider rate-limit retries (no network, no real waiting)."""

import json

import pytest
import requests

import rag_pipeline
from rag_pipeline import MAX_PROVIDER_RETRIES, MAX_RETRY_WAIT_SECONDS


def fake_response(status, body=None, retry_after=None):
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(body or {}).encode()
    if retry_after is not None:
        response.headers["retry-after"] = retry_after
    return response


@pytest.fixture
def fake_http(monkeypatch):
    """Replace requests.post and sleep. Returns (queue of responses, posts, sleeps)."""
    queue, posts, sleeps = [], [], []

    def fake_post(url, headers=None, json=None, timeout=None):
        posts.append(url)
        return queue.pop(0)

    monkeypatch.setattr(rag_pipeline.requests, "post", fake_post)
    monkeypatch.setattr(rag_pipeline, "_sleep", sleeps.append)
    return queue, posts, sleeps


GROQ_OK = {"choices": [{"message": {"content": "the answer"}}]}


def test_retries_after_429_then_succeeds(fake_http):
    queue, posts, sleeps = fake_http
    queue.extend([fake_response(429, retry_after="1"), fake_response(200, GROQ_OK)])
    assert rag_pipeline.ask_groq("prompt") == "the answer"
    assert len(posts) == 2
    assert sleeps == [1.0]


def test_gives_up_after_max_retries(fake_http):
    queue, posts, sleeps = fake_http
    queue.extend([fake_response(429, retry_after="1") for _ in range(MAX_PROVIDER_RETRIES + 1)])
    with pytest.raises(requests.exceptions.HTTPError):
        rag_pipeline.ask_groq("prompt")
    assert len(posts) == MAX_PROVIDER_RETRIES + 1
    assert len(sleeps) == MAX_PROVIDER_RETRIES


def test_other_errors_are_not_retried(fake_http):
    queue, posts, sleeps = fake_http
    queue.append(fake_response(500))
    with pytest.raises(requests.exceptions.HTTPError):
        rag_pipeline.ask_groq("prompt")
    assert len(posts) == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "header, expected",
    [
        ("3", 3.0),                                  # normal value
        ("999", MAX_RETRY_WAIT_SECONDS),             # capped
        ("soon", rag_pipeline.DEFAULT_RETRY_WAIT_SECONDS),  # not a number
        (None, rag_pipeline.DEFAULT_RETRY_WAIT_SECONDS),    # missing header
    ],
)
def test_retry_wait_seconds(header, expected):
    assert rag_pipeline.retry_wait_seconds(fake_response(429, retry_after=header)) == expected