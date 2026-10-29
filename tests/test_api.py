"""API tests for /ask: agent result, limits and error mapping (the agent is faked)."""

import pytest
import requests
from fastapi.testclient import TestClient

import main
import usage_limiter
from agent import DailyCapExceeded, LLMBudgetExceeded
from usage_limiter import MAX_QUESTION_LENGTH, MAX_REQUESTS_PER_MINUTE

TRACE_STEP = {
    "step": "plan", "tool": "llm", "chunks_retrieved": 0, "best_distance": None,
    "retry": 0, "time_ms": 5.0, "llm_calls": 1,
}
GOOD_RESULT = {"answer": "ChromaDB.", "sources": ["a.pdf"], "trace": [TRACE_STEP]}


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(response=response)


@pytest.fixture
def api(monkeypatch):
    """Test client with a fake agent and a fresh rate limiter."""
    usage_limiter._request_times.clear()
    calls = []
    behaviour = {"result": GOOD_RESULT, "error": None}

    def fake_run_agent(question):
        calls.append(question)
        if behaviour["error"] is not None:
            raise behaviour["error"]
        return behaviour["result"]

    monkeypatch.setattr(main, "run_agent", fake_run_agent)
    return TestClient(main.app), calls, behaviour


def ask(client, question="What stores embeddings?"):
    return client.post("/ask", json={"question": question})


def test_ask_returns_answer_sources_and_trace(api):
    client, calls, _ = api
    response = ask(client, "  What stores embeddings?  ")
    assert response.status_code == 200
    assert response.json() == GOOD_RESULT
    assert calls == ["What stores embeddings?"]  # validated and stripped


def test_ask_does_not_charge_daily_cap_upfront(api):
    client, _, _ = api
    before = usage_limiter._daily_counter["count"]
    ask(client)
    assert usage_limiter._daily_counter["count"] == before


@pytest.mark.parametrize("bad", ["", "   ", "x" * (MAX_QUESTION_LENGTH + 1)])
def test_ask_rejects_bad_questions_before_running_agent(api, bad):
    client, calls, _ = api
    assert ask(client, bad).status_code == 400
    assert calls == []


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
def test_ask_maps_errors_to_status_codes(api, error, status):
    client, _, behaviour = api
    behaviour["error"] = error
    response = ask(client)
    assert response.status_code == status
    assert "detail" in response.json()


def test_ask_rate_limit_per_client(api):
    client, _, _ = api
    for _ in range(MAX_REQUESTS_PER_MINUTE):
        assert ask(client).status_code == 200
    assert ask(client).status_code == 429


def test_response_drops_unexpected_trace_fields(api):
    client, _, behaviour = api
    behaviour["result"] = {**GOOD_RESULT, "trace": [{**TRACE_STEP, "prompt": "SECRET_PROMPT"}]}
    response = ask(client)
    assert response.status_code == 200
    assert "SECRET_PROMPT" not in response.text