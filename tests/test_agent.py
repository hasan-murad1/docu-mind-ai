import sys
from pathlib import Path

import pytest

# Make app/core importable (same style as main.py)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app" / "core"))

from agent import (
    AgentTrace,
    LLMBudget,
    LLMBudgetExceeded,
    NO_INFO_ANSWER,
    run_agent,
    validate_question,
)

ALLOWED_KEYS = {
    "step", "tool", "chunks_retrieved", "best_distance",
    "retry", "time_ms", "llm_calls",
}


def fake_prompt(question, chunks):
    return "SECRET_PROMPT_TEXT " + question


def make_search(chunks):
    def _search(question, top_k=7):
        return chunks
    return _search


def test_trace_record_has_expected_fields():
    import time
    trace = AgentTrace()
    trace.record("retrieve", time.perf_counter(), tool="vector_search",
                 chunks_retrieved=3, best_distance=0.12345, llm_calls=0)
    step = trace.to_list()[0]
    assert set(step.keys()) == ALLOWED_KEYS
    assert step["best_distance"] == 0.1235
    assert step["time_ms"] >= 0


def test_budget_blocks_extra_calls():
    budget = LLMBudget(max_calls=2)
    budget.consume()
    budget.consume()
    with pytest.raises(LLMBudgetExceeded):
        budget.consume()


def test_budget_rejects_bad_max():
    with pytest.raises(ValueError):
        LLMBudget(max_calls=0)


@pytest.mark.parametrize("bad", ["", "   ", None, 123, "x" * 501])
def test_validate_question_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_question(bad)


def test_agent_no_relevant_chunks_makes_no_llm_call():
    calls = []
    result = run_agent(
        "anything",
        search_fn=make_search([{"text": "a", "source": "a.pdf", "distance": 5.0}]),
        llm_fn=lambda p: calls.append(p) or "x",
        build_prompt_fn=fake_prompt,
        threshold=2.0,
    )
    assert result["answer"] == NO_INFO_ANSWER
    assert result["sources"] == []
    assert calls == []
    assert result["trace"][-1]["llm_calls"] == 0


def test_agent_happy_path_records_trace_and_sources():
    chunks = [
        {"text": "a", "source": "a.pdf", "distance": 0.5},
        {"text": "b", "source": "b.pdf", "distance": 1.0},
        {"text": "c", "source": "a.pdf", "distance": 9.0},  # filtered out
    ]
    result = run_agent(
        "question",
        search_fn=make_search(chunks),
        llm_fn=lambda p: "the answer",
        build_prompt_fn=fake_prompt,
        threshold=2.0,
    )
    assert result["answer"] == "the answer"
    assert result["sources"] == ["a.pdf", "b.pdf"]
    steps = result["trace"]
    assert [s["step"] for s in steps] == ["retrieve", "generate"]
    assert steps[0]["tool"] == "vector_search"
    assert steps[0]["chunks_retrieved"] == 2
    assert steps[0]["best_distance"] == 0.5
    assert steps[1]["llm_calls"] == 1


def test_trace_never_contains_prompt_or_extra_fields():
    result = run_agent(
        "question",
        search_fn=make_search([{"text": "a", "source": "a.pdf", "distance": 0.5}]),
        llm_fn=lambda p: "ok",
        build_prompt_fn=fake_prompt,
        threshold=2.0,
    )
    assert "SECRET_PROMPT_TEXT" not in str(result["trace"])
    for step in result["trace"]:
        assert set(step.keys()) == ALLOWED_KEYS