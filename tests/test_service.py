"""Tests for the shared service: limits, sessions, upload security and error mapping."""

import pytest
import requests

import usage_limiter
from agent import DailyCapExceeded, LLMBudgetExceeded
from fakes import GOOD_RESULT, SESSION, VALID_PDF, http_error, make_service
from service import ServiceError
from usage_limiter import MAX_QUESTION_LENGTH, MAX_QUESTIONS_PER_DAY, MAX_UPLOADS_PER_DAY

VISITOR = "alice"


@pytest.fixture
def env(tmp_path):
    return make_service(tmp_path)


# ---------- ask ----------

def test_ask_uses_the_visitors_own_collection(env):
    assert env.service.ask(VISITOR, "  What stores embeddings?  ") == GOOD_RESULT
    assert env.agent.calls == ["What stores embeddings?"]  # validated and stripped
    assert env.agent.collections == [SESSION]


@pytest.mark.parametrize("bad", ["", "   ", None, "x" * (MAX_QUESTION_LENGTH + 1)])
def test_bad_questions_are_rejected_without_using_quota(env, bad):
    with pytest.raises(ServiceError) as caught:
        env.service.ask(VISITOR, bad)
    assert caught.value.status == 400
    assert env.agent.calls == []
    assert env.questions.remaining_today(VISITOR) == MAX_QUESTIONS_PER_DAY


@pytest.mark.parametrize(
    "error, status",
    [
        (DailyCapExceeded("cap"), 429),
        (LLMBudgetExceeded("budget"), 429),
        (http_error(429), 429),
        (http_error(500), 502),
        (requests.exceptions.ConnectionError(), 503),
    ],
)
def test_agent_errors_are_mapped_and_refunded(env, error, status):
    env.agent.error = error
    with pytest.raises(ServiceError) as caught:
        env.service.ask(VISITOR, "question")
    assert caught.value.status == status
    assert env.questions.remaining_today(VISITOR) == MAX_QUESTIONS_PER_DAY


def test_unknown_errors_are_raised_and_refunded(env):
    env.agent.error = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        env.service.ask(VISITOR, "question")
    assert env.questions.remaining_today(VISITOR) == MAX_QUESTIONS_PER_DAY


def test_question_daily_limit(env):
    for _ in range(MAX_QUESTIONS_PER_DAY):
        env.service.ask(VISITOR, "question")
    with pytest.raises(ServiceError) as caught:
        env.service.ask(VISITOR, "question")
    assert caught.value.status == 429
    assert "today" in caught.value.message
    env.time.next_day()
    assert env.service.ask(VISITOR, "question") == GOOD_RESULT


def test_ask_does_not_charge_the_global_cap_upfront(env):
    before = usage_limiter._daily_counter["count"]
    env.service.ask(VISITOR, "question")
    assert usage_limiter._daily_counter["count"] == before


# ---------- upload ----------

def test_upload_stores_in_the_visitors_collection(env):
    result = env.service.upload(VISITOR, "notes.pdf", VALID_PDF)
    assert result["chunks_stored"] == 1
    assert result["filename"] == "notes.pdf"
    assert env.stored_in == [("notes.pdf", SESSION)]


def test_upload_cleans_the_filename(env):
    result = env.service.upload(VISITOR, "../../secret report.pdf", VALID_PDF)
    assert result["filename"] == "secret report.pdf"


def test_uploaded_file_is_deleted_after_processing(env):
    env.service.upload(VISITOR, "notes.pdf", VALID_PDF)
    assert list(env.upload_dir.iterdir()) == []


def test_fake_pdf_is_rejected_without_using_quota(env):
    with pytest.raises(ServiceError) as caught:
        env.service.upload(VISITOR, "invoice.pdf", b"MZ\x90\x00 a windows program")
    assert caught.value.status == 400
    assert "real .pdf" in caught.value.message
    assert env.uploads.remaining_today(VISITOR) == MAX_UPLOADS_PER_DAY


def test_unsupported_files_never_use_quota(env):
    for _ in range(5):
        with pytest.raises(ServiceError):
            env.service.upload(VISITOR, "notes.txt", b"plain text")
    assert env.service.upload(VISITOR, "notes.pdf", VALID_PDF)["chunks_stored"] == 1


def test_unreadable_file_is_rejected_and_deleted(env):
    def broken_extract(path):
        raise KeyError("broken document")

    env.service.extract_fn = broken_extract
    with pytest.raises(ServiceError) as caught:
        env.service.upload(VISITOR, "notes.pdf", VALID_PDF)
    assert "couldn't be read" in caught.value.message
    assert list(env.upload_dir.iterdir()) == []
    assert env.uploads.remaining_today(VISITOR) == MAX_UPLOADS_PER_DAY


def test_upload_daily_limit(env):
    for _ in range(MAX_UPLOADS_PER_DAY):
        env.service.upload(VISITOR, "notes.pdf", VALID_PDF)
    with pytest.raises(ServiceError) as caught:
        env.service.upload(VISITOR, "notes.pdf", VALID_PDF)
    assert caught.value.status == 429
    assert "today" in caught.value.message


# ---------- session info ----------

def test_usage_and_documents(env):
    env.service.upload(VISITOR, "notes.pdf", VALID_PDF)
    env.service.ask(VISITOR, "question")
    assert env.service.usage(VISITOR) == {
        "questions_left": MAX_QUESTIONS_PER_DAY - 1,
        "uploads_left": MAX_UPLOADS_PER_DAY - 1,
        "expires_in": 7200,
    }
    assert env.service.documents(VISITOR) == ["notes.pdf"]


def test_end_session(env):
    env.service.documents(VISITOR)
    assert env.service.end_session(VISITOR) is True
    assert env.sessions.ended == [VISITOR]