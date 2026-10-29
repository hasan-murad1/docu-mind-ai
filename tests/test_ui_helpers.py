"""Tests for the pure UI helpers (no Streamlit needed)."""

from ui_helpers import (
    STEP_TEXT,
    escape_markdown,
    sanitize_markdown,
    timeline_lines,
    trace_summary,
)

TRACE = [
    {"step": "plan", "tool": "llm", "chunks_retrieved": 0, "best_distance": None,
     "retry": 0, "time_ms": 676.1, "llm_calls": 1},
    {"step": "retrieve", "tool": "vector_search", "chunks_retrieved": 1, "best_distance": 1.4151,
     "retry": 0, "time_ms": 66.8, "llm_calls": 1},
    {"step": "evaluate_enough", "tool": "llm", "chunks_retrieved": 1, "best_distance": 1.4151,
     "retry": 0, "time_ms": 1241.0, "llm_calls": 2},
    {"step": "draft", "tool": "llm", "chunks_retrieved": 1, "best_distance": None,
     "retry": 0, "time_ms": 616.6, "llm_calls": 3},
    {"step": "self_check_ok", "tool": "llm", "chunks_retrieved": 1, "best_distance": None,
     "retry": 0, "time_ms": 597.2, "llm_calls": 4},
]


def test_sanitize_removes_images_but_keeps_alt_text():
    assert sanitize_markdown("See ![logo](http://evil.example/p.png) here") == "See logo here"


def test_sanitize_keeps_normal_markdown():
    assert sanitize_markdown("It uses **ChromaDB**.") == "It uses **ChromaDB**."


def test_sanitize_non_string_returns_empty():
    assert sanitize_markdown(None) == ""


def test_escape_markdown_makes_filenames_literal():
    assert escape_markdown("my_*file*.pdf") == "my\\_\\*file\\*\\.pdf"


def test_trace_summary():
    assert trace_summary(TRACE) == "5 steps · 4 LLM calls · 3.2s"


def test_trace_summary_empty():
    assert trace_summary([]) == "no steps"


def test_timeline_plan_shows_chosen_tool_and_retrieve_details():
    lines = timeline_lines(TRACE)
    assert len(lines) == 5
    assert "chose vector_search" in lines[0]
    assert "1 chunk · d=1.42" in lines[1]


def test_timeline_shows_retry_number():
    retry_step = {**TRACE[1], "retry": 2}
    assert "retry 2" in timeline_lines([retry_step])[0]


def test_timeline_knows_every_agent_step():
    for name, (icon, label) in STEP_TEXT.items():
        step = {"step": name, "tool": "llm", "chunks_retrieved": 0, "time_ms": 1.0}
        assert timeline_lines([step])[0].startswith(f"{icon} **{label}**")


def test_timeline_handles_unknown_step():
    assert timeline_lines([{"step": "mystery", "time_ms": 1}]) == ["• **mystery** · 1 ms"]