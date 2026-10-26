"""Tests for the agent core: trace, budget, daily cap, planning, evaluate + retry loop."""

import json
import time

import pytest

from agent import (
    MAX_CONTEXT_CHUNKS,
    MAX_LLM_CALLS_PER_QUESTION,
    MAX_RETRIES,
    AgentTrace,
    DailyCapExceeded,
    LLMBudget,
    LLMBudgetExceeded,
    NO_INFO_ANSWER,
    default_tools,
    make_counted_llm,
    merge_chunks,
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

EVAL_ENOUGH = '{"enough": true}'
EVAL_NO_REWRITE = '{"enough": false, "rewrite": null}'


def not_enough(rewrite):
    return json.dumps({"enough": False, "rewrite": rewrite})


CHUNK_A = {"text": "a", "source": "a.pdf", "distance": 0.5}
CHUNK_B = {"text": "b", "source": "b.pdf", "distance": 1.0}
NEAR = [CHUNK_A, CHUNK_B, {"text": "c", "source": "a.pdf", "distance": 9.0}]  # c is filtered out
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
    """
    Fake tools with the real signature. `results` maps tool name -> chunks,
    or tool name -> {query: chunks} when different queries need different results.
    """
    results = results or {}
    calls = []

    def make(name):
        def tool(query, top_k, source=None):
            calls.append({"tool": name, "query": query, "source": source})
            value = results.get(name, [])
            if isinstance(value, dict):
                value = value.get(query, [])
            return list(value)
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
        llm_fn=llm or ScriptedLLM(PLAN_VECTOR, EVAL_ENOUGH, "the answer"),
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


# ---------- Planning + tool choice ----------

def test_agent_no_documents_makes_no_llm_call():
    llm = ScriptedLLM()
    cap = CapCounter()
    tools, calls = make_tools()
    result = run(llm=llm, tools=tools, sources=(), cap=cap)
    assert result["answer"] == NO_INFO_ANSWER
    assert result["sources"] == []
    assert steps(result) == ["finish_no_docs"]
    assert llm.prompts == [] and calls == [] and cap.count == 0


def test_agent_no_relevant_chunks_never_generates():
    llm = ScriptedLLM(PLAN_VECTOR, EVAL_NO_REWRITE)
    tools, _ = make_tools({"vector_search": FAR})
    result = run(llm=llm, tools=tools)
    assert result["answer"] == NO_INFO_ANSWER
    assert result["sources"] == []
    assert steps(result) == ["plan", "retrieve", "evaluate_not_enough", "finish_no_info"]
    assert len(llm.prompts) == 2  # plan + evaluate, no generation


def test_agent_happy_path_records_trace_and_sources():
    tools, _ = make_tools({"vector_search": NEAR})
    result = run(tools=tools)
    trace = result["trace"]
    assert result["answer"] == "the answer"
    assert result["sources"] == ["a.pdf", "b.pdf"]
    assert steps(result) == ["plan", "retrieve", "evaluate_enough", "generate"]
    assert trace[1]["tool"] == "vector_search"
    assert trace[1]["chunks_retrieved"] == 2
    assert trace[1]["best_distance"] == 0.5
    assert trace[2]["tool"] == "llm"
    assert trace[2]["chunks_retrieved"] == 2
    assert trace[3]["llm_calls"] == 3


def test_agent_uses_keyword_tool_chosen_by_planner():
    tools, calls = make_tools({"keyword_search": NEAR})
    result = run(llm=ScriptedLLM(PLAN_KEYWORD, EVAL_ENOUGH, "the answer"), tools=tools)
    assert calls == [{"tool": "keyword_search", "query": "INV-42", "source": None}]
    assert result["trace"][1]["tool"] == "keyword_search"


def test_agent_passes_source_to_metadata_tool():
    tools, calls = make_tools({"metadata_search": NEAR})
    run(llm=ScriptedLLM(PLAN_METADATA, EVAL_ENOUGH, "the answer"), tools=tools)
    assert calls == [{"tool": "metadata_search", "query": "leave days", "source": "b.pdf"}]


def test_agent_invalid_plan_falls_back_to_vector_search():
    tools, calls = make_tools({"vector_search": NEAR})
    result = run(
        question="What is the refund policy?",
        llm=ScriptedLLM("not json at all", EVAL_ENOUGH, "the answer"),
        tools=tools,
    )
    assert steps(result) == ["plan_fallback", "retrieve", "evaluate_enough", "generate"]
    assert calls == [{"tool": "vector_search", "query": "What is the refund policy?", "source": None}]


def test_agent_uses_vector_fallback_when_chosen_tool_finds_nothing():
    tools, calls = make_tools({"keyword_search": FAR, "vector_search": NEAR})
    result = run(
        question="original question",
        llm=ScriptedLLM(PLAN_KEYWORD, EVAL_ENOUGH, "the answer"),
        tools=tools,
    )
    assert steps(result) == ["plan", "retrieve", "retrieve_fallback", "evaluate_enough", "generate"]
    assert calls[1] == {"tool": "vector_search", "query": "original question", "source": None}
    assert result["answer"] == "the answer"


def test_agent_answers_the_original_question_not_the_search_query():
    llm = ScriptedLLM(PLAN_VECTOR, EVAL_ENOUGH, "the answer")
    tools, _ = make_tools({"vector_search": NEAR})
    run(question="original question", llm=llm, tools=tools)
    assert llm.prompts[2] == "SECRET_PROMPT_TEXT original question"


def test_trace_never_contains_prompt_or_extra_fields():
    tools, _ = make_tools({"vector_search": NEAR})
    result = run(tools=tools)
    text = str(result["trace"])
    assert "SECRET_PROMPT_TEXT" not in text
    assert "retrieval planner" not in text
    assert "retrieved document text" not in text
    for step in result["trace"]:
        assert set(step.keys()) == ALLOWED_KEYS


def test_run_agent_counts_every_llm_call_toward_daily_cap():
    cap = CapCounter()
    tools, _ = make_tools({"vector_search": NEAR})
    run(tools=tools, cap=cap)
    assert cap.count == 3  # plan + evaluate + generate


def test_run_agent_stops_before_tools_when_daily_cap_reached():
    llm = ScriptedLLM(PLAN_VECTOR, EVAL_ENOUGH, "the answer")
    tools, calls = make_tools({"vector_search": NEAR})
    with pytest.raises(DailyCapExceeded):
        run(llm=llm, tools=tools, cap=CapCounter(limit=0))
    assert llm.prompts == [] and calls == []


def test_run_agent_daily_cap_reached_at_generate_step():
    llm = ScriptedLLM(PLAN_VECTOR, EVAL_ENOUGH, "the answer")
    tools, _ = make_tools({"vector_search": NEAR})
    with pytest.raises(DailyCapExceeded):
        run(llm=llm, tools=tools, cap=CapCounter(limit=2))
    assert len(llm.prompts) == 2  # plan + evaluate ran, generation was blocked


def test_run_tool_rejects_unknown_tool():
    tools, _ = make_tools()
    with pytest.raises(ValueError):
        run_tool(tools, "delete_everything", "q", 3)


def test_default_tools_match_allowed_tools():
    assert set(default_tools().keys()) == set(ALLOWED_TOOLS)


# ---------- Phase 5: evaluate + rewrite loop ----------

def test_rewrite_then_success():
    llm = ScriptedLLM(PLAN_VECTOR, not_enough("better query"), EVAL_ENOUGH, "the answer")
    tools, calls = make_tools({"vector_search": {"short query": FAR, "better query": NEAR}})
    result = run(llm=llm, tools=tools)
    assert steps(result) == [
        "plan", "retrieve", "evaluate_not_enough", "retrieve", "evaluate_enough", "generate",
    ]
    assert result["trace"][3]["retry"] == 1
    assert calls[1] == {"tool": "vector_search", "query": "better query", "source": None}
    assert result["answer"] == "the answer"


def test_chunks_are_kept_across_retries():
    llm = ScriptedLLM(PLAN_VECTOR, not_enough("better query"), EVAL_ENOUGH, "the answer")
    tools, _ = make_tools({"vector_search": {"short query": [CHUNK_A], "better query": [CHUNK_B]}})
    result = run(llm=llm, tools=tools)
    assert result["sources"] == ["a.pdf", "b.pdf"]
    assert result["trace"][4]["step"] == "evaluate_enough"
    assert result["trace"][4]["chunks_retrieved"] == 2


def test_loop_stops_after_max_retries():
    rewrites = [not_enough(f"q{i}") for i in range(1, MAX_RETRIES + 2)]
    llm = ScriptedLLM(PLAN_VECTOR, *rewrites)  # evaluator never says "enough"
    tools, calls = make_tools({"vector_search": NEAR})
    result = run(llm=llm, tools=tools)
    assert result["answer"] == NO_INFO_ANSWER
    assert result["sources"] == []
    assert len(calls) == MAX_RETRIES + 1                 # first try + 3 retries
    assert len(llm.prompts) == 1 + (MAX_RETRIES + 1)     # plan + one evaluate per try
    assert max(s["retry"] for s in result["trace"]) == MAX_RETRIES
    assert steps(result)[-1] == "finish_no_info"


def test_loop_stops_when_rewrite_repeats_a_previous_query():
    llm = ScriptedLLM(PLAN_VECTOR, not_enough("  Short   QUERY "))
    tools, calls = make_tools({"vector_search": NEAR})
    result = run(llm=llm, tools=tools)
    assert result["answer"] == NO_INFO_ANSWER
    assert len(calls) == 1
    assert len(llm.prompts) == 2


def test_close_but_irrelevant_chunks_give_no_info_and_no_sources():
    # Known issue from earlier: distance passed the threshold but the chunks were
    # off-topic. The evaluator now says "not enough", so sources stay hidden.
    llm = ScriptedLLM(PLAN_VECTOR, EVAL_NO_REWRITE)
    tools, calls = make_tools({"vector_search": NEAR})
    result = run(llm=llm, tools=tools)
    assert result["answer"] == NO_INFO_ANSWER
    assert result["sources"] == []
    assert len(calls) == 1


def test_invalid_evaluator_output_falls_back_to_generating():
    llm = ScriptedLLM(PLAN_VECTOR, "garbage", "the answer")
    tools, _ = make_tools({"vector_search": NEAR})
    result = run(llm=llm, tools=tools)
    assert steps(result) == ["plan", "retrieve", "evaluate_fallback", "generate"]
    assert result["answer"] == "the answer"


def test_empty_final_attempt_spends_no_llm_call():
    llm = ScriptedLLM(PLAN_VECTOR, not_enough("q1"), not_enough("q2"), not_enough("q3"))
    tools, calls = make_tools({"vector_search": FAR})  # nothing is ever relevant
    result = run(llm=llm, tools=tools)
    assert len(calls) == MAX_RETRIES + 1
    assert len(llm.prompts) == 4  # plan + 3 evaluations, none for the last empty try
    assert steps(result)[-2:] == ["retrieve", "finish_no_info"]


def test_retries_use_vector_search_even_if_planner_chose_keyword():
    llm = ScriptedLLM(PLAN_KEYWORD, not_enough("better query"), EVAL_ENOUGH, "the answer")
    tools, calls = make_tools({"keyword_search": NEAR, "vector_search": {"better query": NEAR}})
    run(llm=llm, tools=tools)
    assert calls[0]["tool"] == "keyword_search"
    assert calls[1] == {"tool": "vector_search", "query": "better query", "source": None}


def test_worst_case_stays_within_llm_budget():
    llm = ScriptedLLM(
        PLAN_VECTOR, not_enough("q1"), not_enough("q2"), not_enough("q3"),
        EVAL_ENOUGH, "the answer",
    )
    tools, _ = make_tools({"vector_search": NEAR})
    result = run(llm=llm, tools=tools)
    assert result["answer"] == "the answer"
    assert result["trace"][-1]["llm_calls"] == 6
    assert result["trace"][-1]["llm_calls"] <= MAX_LLM_CALLS_PER_QUESTION


def test_merge_chunks_dedupes_sorts_and_caps():
    better_a = {"text": "a", "source": "a.pdf", "distance": 0.3}
    merged = merge_chunks([CHUNK_A], [better_a, CHUNK_B])
    assert merged == [better_a, CHUNK_B]  # duplicate removed, best distance kept

    many = [{"text": str(i), "source": "x.pdf", "distance": float(i)} for i in range(20, 0, -1)]
    capped = merge_chunks([], many)
    assert len(capped) == MAX_CONTEXT_CHUNKS
    assert [c["distance"] for c in capped] == sorted(c["distance"] for c in capped)