"""Pure helper functions for the Streamlit UI (no Streamlit import, easy to test)."""

import re

# Icon and label for every trace step the agent can produce
STEP_TEXT = {
    "plan": ("🧭", "Plan"),
    "plan_fallback": ("🧭", "Plan"),
    "retrieve": ("🔍", "Retrieve"),
    "retrieve_fallback": ("🔁", "Retrieve (fallback)"),
    "evaluate_enough": ("🧪", "Evaluate"),
    "evaluate_not_enough": ("🧪", "Evaluate"),
    "evaluate_fallback": ("🧪", "Evaluate"),
    "draft": ("✍️", "Draft"),
    "self_check_ok": ("✅", "Self-check"),
    "self_check_revised": ("✏️", "Self-check"),
    "self_check_unsupported": ("⚠️", "Self-check"),
    "self_check_fallback": ("✅", "Self-check"),
    "finish_no_info": ("🚫", "Finish"),
    "finish_no_docs": ("📭", "Finish"),
}

# Steps whose detail text never changes
FIXED_DETAILS = {
    "plan_fallback": "invalid plan → vector_search",
    "evaluate_enough": "context is enough",
    "evaluate_not_enough": "not enough → rewrite or stop",
    "evaluate_fallback": "unclear → continue",
    "self_check_ok": "answer supported",
    "self_check_revised": "answer corrected",
    "self_check_unsupported": "answer not supported",
    "self_check_fallback": "unclear → kept draft",
    "finish_no_info": "no information found",
    "finish_no_docs": "no documents uploaded",
}

# ![alt](url) -> markdown image. Images are removed so a document can't
# smuggle in a tracking pixel or leak data through an image URL.
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


def sanitize_markdown(text) -> str:
    """Make model or user text safe to render as markdown (images become plain alt text)."""
    if not isinstance(text, str):
        return ""
    return _MD_IMAGE.sub(r"\1", text)


def escape_markdown(text) -> str:
    """Show text (e.g. a filename) literally, with no markdown formatting."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def trace_summary(trace) -> str:
    """One-line summary, e.g. '5 steps · 4 LLM calls · 3.2s'."""
    if not trace:
        return "no steps"
    total_ms = sum(float(step.get("time_ms") or 0) for step in trace)
    llm_calls = trace[-1].get("llm_calls", 0)
    return f"{len(trace)} steps · {llm_calls} LLM calls · {total_ms / 1000:.1f}s"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _step_detail(step: dict, next_tool) -> str:
    name = step.get("step", "")
    if name == "plan":
        return f"chose {next_tool}" if next_tool else "plan ready"
    if name in ("retrieve", "retrieve_fallback"):
        detail = f"{step.get('tool')} · {_plural(step.get('chunks_retrieved', 0), 'chunk')}"
        if step.get("best_distance") is not None:
            detail += f" · d={step['best_distance']:.2f}"
        if step.get("retry"):
            detail += f" · retry {step['retry']}"
        return detail
    if name == "draft":
        return f"written from {_plural(step.get('chunks_retrieved', 0), 'chunk')}"
    return FIXED_DETAILS.get(name, "")


def timeline_lines(trace) -> list[str]:
    """Turn the trace into readable markdown lines, one per step."""
    trace = trace or []
    lines = []
    for i, step in enumerate(trace):
        name = str(step.get("step", "?"))
        icon, label = STEP_TEXT.get(name, ("•", name))
        # The plan step's real choice is the tool used by the next retrieve step
        next_tool = next(
            (s.get("tool") for s in trace[i + 1:] if str(s.get("step", "")).startswith("retrieve")),
            None,
        )
        detail = _step_detail(step, next_tool)
        ms = float(step.get("time_ms") or 0)
        if detail:
            lines.append(f"{icon} **{label}** — {detail} · {ms:.0f} ms")
        else:
            lines.append(f"{icon} **{label}** · {ms:.0f} ms")
    return lines