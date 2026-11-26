"""Tests for the Streamlit visitor helpers."""

import pytest

from session_helpers import SAMPLE_QUESTIONS, client_ip_from_headers, format_duration, usage_line
from usage_limiter import MAX_QUESTION_LENGTH


@pytest.mark.parametrize(
    "headers, expected",
    [
        ({"X-Forwarded-For": "203.0.113.5, 10.0.0.1"}, "203.0.113.5"),
        ({"x-forwarded-for": " 2001:db8::1 "}, "2001:db8::1"),
        ({"X-Forwarded-For": "not-an-ip"}, None),
        ({}, None),
        (None, None),
    ],
)
def test_client_ip_from_headers(headers, expected):
    assert client_ip_from_headers(headers) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [(7140, "1h 59m"), (3600, "1h 0m"), (600, "10m"), (30, "less than a minute"), (None, "")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_usage_line():
    line = usage_line({"questions_left": 4, "uploads_left": 1, "expires_in": 6720})
    assert line == "4 questions left today · 1 upload left · files deleted in 1h 52m"


def test_usage_line_without_session():
    line = usage_line({"questions_left": 5, "uploads_left": 3, "expires_in": None})
    assert line == "5 questions left today · 3 uploads left"


def test_sample_questions_are_valid():
    assert SAMPLE_QUESTIONS
    assert all(0 < len(q) <= MAX_QUESTION_LENGTH for q in SAMPLE_QUESTIONS)