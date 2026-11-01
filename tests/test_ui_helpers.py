"""Tests for the pure UI helpers (no Streamlit needed)."""

import pytest

from ui_helpers import (
    MAX_HISTORY,
    STEP_LABELS,
    THEMES,
    chat_title,
    css_variables,
    escape_markdown,
    find_chat,
    format_ms,
    is_no_info,
    sanitize_markdown,
    sources_html,
    timeline_html,
    timeline_rows,
    trace_summary,
    upsert_chat,
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

NO_INFO = "I don't have information about that in the uploaded document(s)."


def user(text):
    return {"role": "user", "content": text}


def bot(text):
    return {"role": "assistant", "content": text}


# ---------- Text safety ----------

def test_sanitize_removes_images_but_keeps_alt_text():
    assert sanitize_markdown("See ![logo](http://evil.example/p.png) here") == "See logo here"


def test_sanitize_keeps_normal_markdown():
    assert sanitize_markdown("It uses **ChromaDB**.") == "It uses **ChromaDB**."


def test_sanitize_non_string_returns_empty():
    assert sanitize_markdown(None) == ""


def test_escape_markdown_makes_text_literal():
    assert escape_markdown("my_*file*.pdf") == "my\\_\\*file\\*\\.pdf"


@pytest.mark.parametrize(
    "text, expected",
    [
        (NO_INFO, True),
        ("I don\u2019t have information about the CEO.", True),
        ("", True),
        (None, True),
        ("The CEO is Rafiq Ahmed.", False),
    ],
)
def test_is_no_info(text, expected):
    assert is_no_info(text) is expected


def test_sources_html_escapes_filenames():
    out = sources_html(["<script>x</script>.pdf"])
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_sources_html_empty():
    assert sources_html([]) == ""


# ---------- Theme ----------

def test_css_variables_differ_per_theme():
    dark, light = css_variables("dark"), css_variables("light")
    assert "--dm-bg:#0B0B0C;" in dark
    assert "--dm-bg:#FAFAF7;" in light


def test_unknown_theme_falls_back_to_dark():
    assert css_variables("purple") == css_variables("dark")


def test_every_theme_has_the_same_tokens():
    assert set(THEMES["dark"]) == set(THEMES["light"])


# ---------- Agent trace ----------

@pytest.mark.parametrize("ms, expected", [(268.0, "268 ms"), (1420.0, "1.4 s"), (None, "0 ms")])
def test_format_ms(ms, expected):
    assert format_ms(ms) == expected


def test_trace_summary():
    assert trace_summary(TRACE) == "5 steps · 4 LLM calls · 3.2s"


def test_trace_summary_empty():
    assert trace_summary([]) == "no steps"


def test_timeline_plan_shows_chosen_tool_and_retrieve_details():
    rows = timeline_rows(TRACE)
    assert len(rows) == 5
    assert rows[0]["detail"] == "chose vector_search"
    assert rows[1]["detail"] == "1 chunk · d=1.42"
    assert rows[1]["time"] == "67 ms"


def test_timeline_shows_retry_number():
    assert "retry 2" in timeline_rows([{**TRACE[1], "retry": 2}])[0]["detail"]


def test_timeline_tones():
    rows = timeline_rows(TRACE + [{"step": "finish_no_info", "time_ms": 0}])
    assert rows[1]["tone"] == "accent"
    assert rows[4]["tone"] == "ok"
    assert rows[5]["tone"] == "warn"


def test_timeline_knows_every_agent_step():
    for name, label in STEP_LABELS.items():
        step = {"step": name, "tool": "llm", "chunks_retrieved": 0, "time_ms": 1.0}
        assert timeline_rows([step])[0]["label"] == label


def test_timeline_handles_unknown_step():
    row = timeline_rows([{"step": "mystery", "time_ms": 1}])[0]
    assert (row["label"], row["detail"], row["time"]) == ("mystery", "", "1 ms")


def test_timeline_html_escapes_values():
    out = timeline_html([{"step": "<b>x</b>", "time_ms": 1}])
    assert "<b>x</b>" not in out
    assert "&lt;b&gt;x&lt;/b&gt;" in out


# ---------- Chat history ----------

def test_chat_title_uses_first_question_and_truncates():
    title = chat_title([user("  a" * 30), bot("answer")])
    assert len(title) == 40
    assert title.endswith("…")


def test_chat_title_without_question():
    assert chat_title([]) == "New chat"


def test_upsert_chat_adds_newest_first():
    chats = upsert_chat([], "a", [user("first")])
    chats = upsert_chat(chats, "b", [user("second")])
    assert [c["id"] for c in chats] == ["b", "a"]


def test_upsert_chat_updates_existing_and_moves_it_to_top():
    chats = upsert_chat([], "a", [user("first")])
    chats = upsert_chat(chats, "b", [user("second")])
    chats = upsert_chat(chats, "a", [user("first"), bot("answer")])
    assert [c["id"] for c in chats] == ["a", "b"]
    assert len(chats[0]["messages"]) == 2


def test_upsert_chat_skips_empty_chats():
    assert upsert_chat([], "a", []) == []


def test_upsert_chat_keeps_at_most_max_history():
    chats = []
    for i in range(MAX_HISTORY + 5):
        chats = upsert_chat(chats, str(i), [user(f"q{i}")])
    assert len(chats) == MAX_HISTORY
    assert chats[0]["id"] == str(MAX_HISTORY + 4)


def test_find_chat():
    chats = upsert_chat([], "a", [user("first")])
    assert find_chat(chats, "a")["title"] == "first"
    assert find_chat(chats, "missing") is None