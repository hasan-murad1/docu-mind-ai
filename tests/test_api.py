"""API tests for /ask and /upload: agent result, per-visitor limits, upload security
and error mapping. The agent and the vector store are faked, so no model or network is used.
"""

import io
from datetime import date
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient
from pypdf import PdfWriter

import main
import usage_limiter
from agent import DailyCapExceeded, LLMBudgetExceeded
from usage_limiter import (
    MAX_QUESTION_LENGTH,
    MAX_QUESTIONS_PER_DAY,
    MAX_QUESTIONS_PER_MINUTE,
    MAX_UPLOADS_PER_DAY,
    MAX_UPLOADS_PER_MINUTE,
    VisitorLimiter,
)

VISITOR = "testclient"  # the client host that FastAPI's TestClient reports

TRACE_STEP = {
    "step": "plan", "tool": "llm", "chunks_retrieved": 0, "best_distance": None,
    "retry": 0, "time_ms": 5.0, "llm_calls": 1,
}
GOOD_RESULT = {"answer": "ChromaDB.", "sources": ["a.pdf"], "trace": [TRACE_STEP]}


def make_pdf(pages=1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


VALID_PDF = make_pdf()


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(response=response)


class FakeTime:
    """Controllable clock and calendar day for the limiters."""

    def __init__(self):
        self.now = 1000.0
        self.day = date(2026, 11, 17)

    def clock(self):
        return self.now

    def today(self):
        return self.day


@pytest.fixture
def api(monkeypatch, tmp_path):
    """Test client with a fake agent, fake storage and fresh per-visitor limiters."""
    fake_time = FakeTime()
    questions = VisitorLimiter(MAX_QUESTIONS_PER_MINUTE, MAX_QUESTIONS_PER_DAY,
                               clock=fake_time.clock, today=fake_time.today)
    uploads = VisitorLimiter(MAX_UPLOADS_PER_MINUTE, MAX_UPLOADS_PER_DAY,
                             clock=fake_time.clock, today=fake_time.today)
    monkeypatch.setattr(main, "question_limiter", questions)
    monkeypatch.setattr(main, "upload_limiter", uploads)
    monkeypatch.setattr(main, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(main, "extract_text", lambda path: "some readable text " * 20)
    monkeypatch.setattr(main, "add_chunks_to_store", lambda chunks, source_filename: len(chunks))

    calls = []
    behaviour = {"result": GOOD_RESULT, "error": None}

    def fake_run_agent(question):
        calls.append(question)
        if behaviour["error"] is not None:
            raise behaviour["error"]
        return behaviour["result"]

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    return SimpleNamespace(
        client=TestClient(main.app), calls=calls, behaviour=behaviour,
        time=fake_time, questions=questions, uploads=uploads, upload_dir=tmp_path,
    )


def ask(client, question="What stores embeddings?"):
    return client.post("/ask", json={"question": question})


def upload(client, name="notes.pdf", content=VALID_PDF):
    return client.post("/upload", files={"file": (name, content)})


# ---------- /ask ----------

def test_ask_returns_answer_sources_and_trace(api):
    response = ask(api.client, "  What stores embeddings?  ")
    assert response.status_code == 200
    assert response.json() == GOOD_RESULT
    assert api.calls == ["What stores embeddings?"]  # validated and stripped


def test_ask_does_not_charge_daily_cap_upfront(api):
    before = usage_limiter._daily_counter["count"]
    ask(api.client)
    assert usage_limiter._daily_counter["count"] == before


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_QUESTION_LENGTH + 1)])
def test_ask_rejects_bad_questions_without_using_quota(api, bad):
    assert ask(api.client, bad).status_code == 400
    assert api.calls == []
    assert api.questions.remaining_today(VISITOR) == MAX_QUESTIONS_PER_DAY


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
def test_ask_maps_errors_and_gives_the_question_back(api, error, status):
    api.behaviour["error"] = error
    response = ask(api.client)
    assert response.status_code == status
    assert "detail" in response.json()
    assert api.questions.remaining_today(VISITOR) == MAX_QUESTIONS_PER_DAY


def test_ask_daily_limit_without_waiting(api):
    # All daily questions can be asked back to back (no per-minute limit)
    for _ in range(MAX_QUESTIONS_PER_DAY):
        assert ask(api.client).status_code == 200
    blocked = ask(api.client)
    assert blocked.status_code == 429
    assert "today" in blocked.json()["detail"]
    # Next day: both the date and the clock move forward, like in real life
    api.time.day = date(2026, 11, 18)
    api.time.now += 24 * 60 * 60
    assert ask(api.client).status_code == 200


def test_response_drops_unexpected_trace_fields(api):
    api.behaviour["result"] = {**GOOD_RESULT, "trace": [{**TRACE_STEP, "prompt": "SECRET_PROMPT"}]}
    response = ask(api.client)
    assert response.status_code == 200
    assert "SECRET_PROMPT" not in response.text


# ---------- /upload ----------

def test_upload_valid_file(api):
    response = upload(api.client)
    assert response.status_code == 200
    assert response.json()["chunks_stored"] == 1


def test_upload_returns_a_clean_filename(api):
    response = upload(api.client, name="../../secret report.pdf")
    assert response.status_code == 200
    assert response.json()["filename"] == "secret report.pdf"


def test_uploaded_file_is_deleted_after_processing(api):
    assert upload(api.client).status_code == 200
    assert list(api.upload_dir.iterdir()) == []


def test_fake_pdf_is_rejected_without_using_quota(api):
    response = upload(api.client, name="invoice.pdf", content=b"MZ\x90\x00 a windows program")
    assert response.status_code == 400
    assert "real .pdf" in response.json()["detail"]
    assert api.uploads.remaining_today(VISITOR) == MAX_UPLOADS_PER_DAY


def test_upload_daily_limit_without_waiting(api):
    for _ in range(MAX_UPLOADS_PER_DAY):
        assert upload(api.client).status_code == 200
    blocked = upload(api.client)
    assert blocked.status_code == 429
    assert "today" in blocked.json()["detail"]


def test_rejected_upload_does_not_use_quota(api):
    for _ in range(5):
        assert upload(api.client, name="notes.txt").status_code == 400
    assert api.uploads.remaining_today(VISITOR) == MAX_UPLOADS_PER_DAY
    assert upload(api.client).status_code == 200