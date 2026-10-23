"""Tests for the agent core: trace, budget, daily cap, planning and retrieval."""

import time

import pytest

from agent import (
    AgentTrace,
    DailyCapExceeded,
    LLMBudget,
    LLMBudgetExceeded,
    NO_INFO_ANSWER,
    default_tools,
    make_counted_llm,
    run_agent,
    run_tool,
    validate_question,
)
from planner import ALLOWED_TOOLS

ALLOWED_KEYS = {
    "step", "tool", "chunks_retrieved", "best_distance",
    "retry", "time_ms", "llm_calls",
}

PLAN_VECTOR = '{"tool": "vector_search", "query": "short query", "source": null}'
PLAN_KEYWORD = '{"tool": "keyword_search", "query": "INV-42", "source": null}'
PLAN_METADATA = '{"tool": "metadata_search", "query": "leave days", "source": "b.pdf"}'

NEAR = [
    {"text": "a", "source": "a.pdf", "distance": 0.5},
    {"text": "b", "source": "b.pdf", "distance": 1.0},
    {"text": "c", "source": "a.pdf", "distance": 9.0},  # filtered out
]
FAR = [{"text": "far", "source": "a.pdf", "distance": 5.0}]


# ---------- Fakes ----------

def fake_prompt(question, chunks):
    return "SECRET_PROMPT_TEXT " + question


class ScriptedLLM:
    """Returns the given replies in order; fails loudly on an unexpected extra call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if len(self.prompts) > len(self.replies):
            raise AssertionError("Unexpected extra LLM call")
        return self.replies[len(self.prompts) - 1]


class CapCounter:
    """Fake daily cap: allows `limit` calls, records how many were charged."""

    def __init__(self, limit=100):
        self.limit = limit
        self.count = 0

    def __call__(self):
        if self.count >= self.limit:
            return False
        self.count += 1
        return True


def make_tools(results=None):
    """Fake tools with the real signature. `results` maps tool name -> chunks."""
    results = results or {}
    calls = []

    def make(name):
        def tool(query, top_k, source=None):
            calls.append({"tool": name, "query": query, "source": source})
            return list(results.get(name, []))
        return tool

    return {name: make(name) for name in ALLOWED_TOOLS}, calls


def run(question="question", llm=None, tools=None, sources=("a.pdf", "b.pdf"), cap=None):
    """Run the agent with fakes only (no model, no network)."""
    if tools is None:
        tools, _ = make_tools()
    return run_agent(
        question,
        tools=tools,
        sources_fn=lambda: list(sources),
        llm_fn=llm or ScriptedLLM(PLAN_VECTOR, "the answer"),
        build_prompt_fn=fake_prompt,
        threshold=2.0,
        daily_cap_fn=cap or CapCounter(),
    )


def steps(result):
    return [s["step"] for s in result["trace"]]


# ---------- Trace, budget, validation ----------

def test_trace_record_has_expected_fields():
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


# ---------- Counted LLM (daily cap) ----------

def test_counted_llm_charges_daily_cap_for_every_call():
    cap = CapCounter()
    llm = make_counted_llm(LLMBudget(max_calls=5), lambda p: "ok", cap)
    llm("a")
    llm("b")
    assert cap.count == 2


def test_counted_llm_stops_when_daily_cap_reached():
    called = []
    llm = make_counted_llm(LLMBudget(), lambda p: called.append(p) or "ok", CapCounter(limit=0))
    with pytest.raises(DailyCapExceeded):
        llm("a")
    assert called == []


def test_counted_llm_budget_exhausted_does_not_charge_daily_cap():
    cap = CapCounter()
    llm = make_counted_llm(LLMBudget(max_calls=1), lambda p: "ok", cap)
    llm("a")
    with pytest.raises(LLMBudgetExceeded):
        llm("b")
    assert cap.count == 1


def test_call_llm_switches_provider(monkeypatch):
    import rag_pipeline

    monkeypatch.setattr(rag_pipeline, "ask_groq", lambda p: "from_groq")
    monkeypatch.setattr(rag_pipeline, "ask_llm", lambda p: "from_ollama")

    monkeypatch.setattr(rag_pipeline, "LLM_PROVIDER", "groq")
    assert rag_pipeline.call_llm("x") == "from_groq"

    monkeypatch.setattr(rag_pipeline, "LLM_PROVIDER", "ollama")
    assert rag_pipeline.call_llm("x") == "from_ollama"

    monkeypatch.setattr(rag_pipeline, "LLM_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        rag_pipeline.call_llm("x")


# ---------- Phase 4: planning + tool choice ----------

def test_agent_no_documents_makes_no_llm_call():
    llm = ScriptedLLM()
    cap = CapCounter()
    tools, calls = make_tools()
    result = run(llm=llm, tools=tools, sources=(), cap=cap)
    assert result["answer"] == NO_INFO_ANSWER
    assert result["sources"] == []
    assert steps(result) == ["finish_no_docs"]
    assert llm.prompts == [] and calls == [] and cap.count == 0


def test_agent_no_relevant_chunks_skips_generation():
    llm = ScriptedLLM(PLAN_VECTOR)
    tools, _ = make_tools({"vector_search": FAR})
    result = run(llm=llm, tools=tools)
    assert result["answer"] == NO_INFO_ANSWER
    assert result["sources"] == []
    assert steps(result) == ["plan", "retrieve", "finish_no_info"]
    assert len(llm.prompts) == 1  # only the planner call
    assert result["trace"][-1]["llm_calls"] == 1


def test_agent_happy_path_records_trace_and_sources():
    tools, _ = make_tools({"vector_search": NEAR})
    result = run(tools=tools)
    trace = result["trace"]
    assert result["answer"] == "the answer"
    assert result["sources"] == ["a.pdf", "b.pdf"]
    assert steps(result) == ["plan", "retrieve", "generate"]
    assert trace[0]["tool"] == "llm"
    assert trace[1]["tool"] == "vector_search"
    assert trace[1]["chunks_retrieved"] == 2
    assert trace[1]["best_distance"] == 0.5
    assert trace[2]["llm_calls"] == 2


def test_agent_uses_keyword_tool_chosen_by_planner():
    tools, calls = make_tools({"keyword_search": NEAR})
    result = run(llm=ScriptedLLM(PLAN_KEYWORD, "the answer"), tools=tools)
    assert calls == [{"tool": "keyword_search", "query": "INV-42", "source": None}]
    assert result["trace"][1]["tool"] == "keyword_search"


def test_agent_passes_source_to_metadata_tool():
    tools, calls = make_tools({"metadata_search": NEAR})
    run(llm=ScriptedLLM(PLAN_METADATA, "the answer"), tools=tools)
    assert calls == [{"tool": "metadata_search", "query": "leave days", "source": "b.pdf"}]


def test_agent_invalid_plan_falls_back_to_vector_search():
    tools, calls = make_tools({"vector_search": NEAR})
    result = run(
        question="What is the refund policy?",
        llm=ScriptedLLM("not json at all", "the answer"),
        tools=tools,
    )
    assert steps(result) == ["plan_fallback", "retrieve", "generate"]
    assert calls == [{"tool": "vector_search", "query": "What is the refund policy?", "source": None}]


def test_agent_retries_with_vector_search_when_chosen_tool_finds_nothing():
    tools, calls = make_tools({"keyword_search": FAR, "vector_search": NEAR})
    result = run(
        question="original question",
        llm=ScriptedLLM(PLAN_KEYWORD, "the answer"),
        tools=tools,
    )
    assert steps(result) == ["plan", "retrieve", "retrieve_fallback", "generate"]
    assert calls[1] == {"tool": "vector_search", "query": "original question", "source": None}
    assert result["answer"] == "the answer"


def test_agent_answers_the_original_question_not_the_search_query():
    llm = ScriptedLLM(PLAN_VECTOR, "the answer")
    tools, _ = make_tools({"vector_search": NEAR})
    run(question="original question", llm=llm, tools=tools)
    assert llm.prompts[1] == "SECRET_PROMPT_TEXT original question"


def test_trace_never_contains_prompt_or_extra_fields():
    tools, _ = make_tools({"vector_search": NEAR})
    result = run(tools=tools)
    assert "SECRET_PROMPT_TEXT" not in str(result["trace"])
    assert "retrieval planner" not in str(result["trace"])
    for step in result["trace"]:
        assert set(step.keys()) == ALLOWED_KEYS


def test_run_agent_counts_every_llm_call_toward_daily_cap():
    cap = CapCounter()
    tools, _ = make_tools({"vector_search": NEAR})
    run(tools=tools, cap=cap)
    assert cap.count == 2  # planner + generate


def test_run_agent_stops_before_tools_when_daily_cap_reached():
    llm = ScriptedLLM(PLAN_VECTOR, "the answer")
    tools, calls = make_tools({"vector_search": NEAR})
    with pytest.raises(DailyCapExceeded):
        run(llm=llm, tools=tools, cap=CapCounter(limit=0))
    assert llm.prompts == [] and calls == []


def test_run_agent_daily_cap_reached_at_generate_step():
    llm = ScriptedLLM(PLAN_VECTOR, "the answer")
    tools, _ = make_tools({"vector_search": NEAR})
    with pytest.raises(DailyCapExceeded):
        run(llm=llm, tools=tools, cap=CapCounter(limit=1))
    assert len(llm.prompts) == 1  # planner ran, generation was blocked


def test_run_tool_rejects_unknown_tool():
    tools, _ = make_tools()
    with pytest.raises(ValueError):
        run_tool(tools, "delete_everything", "q", 3)


def test_default_tools_match_allowed_tools():
    assert set(default_tools().keys()) == set(ALLOWED_TOOLS)