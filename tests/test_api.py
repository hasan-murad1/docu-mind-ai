"""API tests: HTTP status codes, response shape and the upload size cap.

The flows themselves are tested in test_service.py; here the service runs on fakes.
"""

import pytest
import requests
from fastapi.testclient import TestClient

import main
from agent import DailyCapExceeded, LLMBudgetExceeded
from fakes import GOOD_RESULT, TRACE_STEP, VALID_PDF, http_error, make_service
from upload_security import MAX_UPLOAD_BYTES
from usage_limiter import MAX_QUESTIONS_PER_DAY, MAX_UPLOADS_PER_DAY

VISITOR = "testclient"  # the client host that FastAPI's TestClient reports


@pytest.fixture
def api(monkeypatch, tmp_path):
    env = make_service(tmp_path)
    monkeypatch.setattr(main, "app_service", env.service)
    env.client = TestClient(main.app)
    return env


def ask(client, question="What stores embeddings?"):
    return client.post("/ask", json={"question": question})


def upload(client, name="notes.pdf", content=VALID_PDF):
    return client.post("/upload", files={"file": (name, content)})


def test_ask_returns_answer_sources_and_trace(api):
    response = ask(api.client)
    assert response.status_code == 200
    assert response.json() == GOOD_RESULT
    assert api.sessions.visitors == [VISITOR]  # the visitor is the client host


def test_bad_question_is_400(api):
    assert ask(api.client, "   ").status_code == 400


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
def test_agent_errors_become_http_status_codes(api, error, status):
    api.agent.error = error
    response = ask(api.client)
    assert response.status_code == status
    assert "detail" in response.json()


def test_question_daily_limit_is_429(api):
    for _ in range(MAX_QUESTIONS_PER_DAY):
        assert ask(api.client).status_code == 200
    assert ask(api.client).status_code == 429


def test_response_drops_unexpected_trace_fields(api):
    api.agent.result = {**GOOD_RESULT, "trace": [{**TRACE_STEP, "prompt": "SECRET_PROMPT"}]}
    response = ask(api.client)
    assert response.status_code == 200
    assert "SECRET_PROMPT" not in response.text


def test_upload_valid_file(api):
    response = upload(api.client, name="../../secret report.pdf")
    assert response.status_code == 200
    assert response.json()["filename"] == "secret report.pdf"


def test_fake_pdf_is_400(api):
    response = upload(api.client, name="invoice.pdf", content=b"MZ\x90\x00 a windows program")
    assert response.status_code == 400
    assert "real .pdf" in response.json()["detail"]


def test_too_large_upload_is_400_before_using_quota(api):
    response = upload(api.client, content=b"%PDF-" + b"0" * MAX_UPLOAD_BYTES)
    assert response.status_code == 400
    assert "too large" in response.json()["detail"]
    assert api.uploads.remaining_today(VISITOR) == MAX_UPLOADS_PER_DAY


def test_upload_daily_limit_is_429(api):
    for _ in range(MAX_UPLOADS_PER_DAY):
        assert upload(api.client).status_code == 200
    assert upload(api.client).status_code == 429