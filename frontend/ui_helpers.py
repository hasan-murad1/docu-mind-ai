"""Pure helper functions for the Streamlit UI (no Streamlit import, easy to test)."""

import html
import re

MAX_HISTORY = 20       # chats kept in the sidebar history
MAX_TITLE_CHARS = 40   # history titles are cut to this length

DEFAULT_THEME = "dark"

# Color tokens for each theme. They become CSS variables (--dm-<name>).
THEMES = {
    "dark": {
        "bg": "#0B0B0C", "side": "#111113", "surface": "#18181B",
        "border": "#232327", "border2": "#2E2E33", "text": "#ECECEE",
        "muted": "#A1A1A9", "bubble": "#1E1E22", "input": "#141416",
        "accent": "#F5B544", "ok": "#6EE7A0", "warn": "#F59E54",
    },
    "light": {
        "bg": "#FAFAF7", "side": "#F2F1EC", "surface": "#FFFFFF",
        "border": "#E4E2DC", "border2": "#D3D0C7", "text": "#18181B",
        "muted": "#5C5C63", "bubble": "#ECEAE3", "input": "#FFFFFF",
        "accent": "#F5B544", "ok": "#15803D", "warn": "#C2410C",
    },
}

# Label for every trace step the agent can produce
STEP_LABELS = {
    "plan": "Plan",
    "plan_fallback": "Plan",
    "retrieve": "Retrieve",
    "retrieve_fallback": "Retrieve (fallback)",
    "evaluate_enough": "Evaluate",
    "evaluate_not_enough": "Evaluate",
    "evaluate_fallback": "Evaluate",
    "draft": "Draft",
    "self_check_ok": "Self-check",
    "self_check_revised": "Self-check",
    "self_check_unsupported": "Self-check",
    "self_check_fallback": "Self-check",
    "finish_no_info": "Finish",
    "finish_no_docs": "Finish",
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

# Dot color for a step: "ok" (green), "warn" (orange), otherwise the accent
OK_STEPS = {"self_check_ok", "self_check_revised"}
WARN_STEPS = {
    "plan_fallback", "evaluate_not_enough", "evaluate_fallback",
    "self_check_unsupported", "self_check_fallback",
    "finish_no_info", "finish_no_docs",
}

_NO_INFO_PREFIXES = ("i don't have information", "i do not have information")

# ![alt](url) -> markdown image. Images are removed so a document can't
# smuggle in a tracking pixel or leak data through an image URL.
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~])")


# ---------- Theme ----------

def theme_tokens(theme) -> dict:
    """Color tokens for a theme name; unknown names fall back to the default theme."""
    return THEMES.get(theme, THEMES[DEFAULT_THEME])


def css_variables(theme) -> str:
    """CSS that defines one --dm-* variable per color token."""
    tokens = theme_tokens(theme)
    body = "".join(f"--dm-{name}:{value};" for name, value in tokens.items())
    return f":root{{{body}}}"


# ---------- Text safety ----------

def sanitize_markdown(text) -> str:
    """Make model or user text safe to render as markdown (images become plain alt text)."""
    if not isinstance(text, str):
        return ""
    return _MD_IMAGE.sub(r"\1", text)


def escape_markdown(text) -> str:
    """Show text (e.g. a chat title on a button) literally, with no markdown formatting."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def is_no_info(text) -> bool:
    """True if an answer is empty or is the standard 'no information' reply."""
    if not isinstance(text, str):
        return True
    normalized = " ".join(text.replace("\u2019", "'").lower().split()).lstrip("\"'*` ")
    return not normalized or normalized.startswith(_NO_INFO_PREFIXES)


def sources_html(sources) -> str:
    """Source chips as HTML. Filenames come from users, so they are escaped."""
    chips = "".join(
        f'<span class="dm-chip">{html.escape(str(s))}</span>' for s in (sources or [])
    )
    return f'<div class="dm-sources">{chips}</div>' if chips else ""


# ---------- Agent trace ----------

def format_ms(ms) -> str:
    """'268 ms' below one second, '1.4 s' above."""
    try:
        value = float(ms or 0)
    except (TypeError, ValueError):
        value = 0.0
    return f"{value:.0f} ms" if value < 1000 else f"{value / 1000:.1f} s"


def trace_summary(trace) -> str:
    """One-line summary, e.g. '5 steps · 4 LLM calls · 3.2s'."""
    if not trace:
        return "no steps"
    total_ms = 0.0
    for step in trace:
        try:
            total_ms += float(step.get("time_ms") or 0)
        except (TypeError, ValueError):
            pass
    llm_calls = trace[-1].get("llm_calls", 0)
    return f"{len(trace)} steps · {llm_calls} LLM calls · {total_ms / 1000:.1f}s"


def _plural(n, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _step_detail(step: dict, next_tool) -> str:
    name = step.get("step", "")
    if name == "plan":
        return f"chose {next_tool}" if next_tool else "plan ready"
    if name in ("retrieve", "retrieve_fallback"):
        detail = _plural(step.get("chunks_retrieved", 0), "chunk")
        distance = step.get("best_distance")
        if isinstance(distance, (int, float)) and not isinstance(distance, bool):
            detail += f" · d={distance:.2f}"
        if step.get("retry"):
            detail += f" · retry {step['retry']}"
        return detail
    if name == "draft":
        return f"from {_plural(step.get('chunks_retrieved', 0), 'chunk')}"
    return FIXED_DETAILS.get(name, "")


def timeline_rows(trace) -> list[dict]:
    """Turn the trace into rows: label, detail, time and dot tone."""
    trace = trace or []
    rows = []
    for i, step in enumerate(trace):
        name = str(step.get("step", "?"))
        # The plan step's real choice is the tool used by the next retrieve step
        next_tool = next(
            (s.get("tool") for s in trace[i + 1:] if str(s.get("step", "")).startswith("retrieve")),
            None,
        )
        if name in OK_STEPS:
            tone = "ok"
        elif name in WARN_STEPS:
            tone = "warn"
        else:
            tone = "accent"
        rows.append({
            "label": STEP_LABELS.get(name, name),
            "detail": _step_detail(step, next_tool),
            "time": format_ms(step.get("time_ms")),
            "tone": tone,
        })
    return rows


def timeline_html(trace) -> str:
    """The agent timeline as an HTML list. Every value is escaped."""
    items = []
    for row in timeline_rows(trace):
        items.append(
            "<li>"
            f'<span class="dm-dot dm-dot-{row["tone"]}"></span>'
            f"<span>{html.escape(row['label'])}</span>"
            f'<span class="dm-muted">{html.escape(row["detail"])}</span>'
            f'<span class="dm-muted">{html.escape(row["time"])}</span>'
            "</li>"
        )
    return '<ol class="dm-steps">' + "".join(items) + "</ol>"


# ---------- Chat history (kept in the browser session only) ----------

def chat_title(messages, max_len: int = MAX_TITLE_CHARS) -> str:
    """Title of a chat: its first user question, shortened."""
    for message in messages or []:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            title = " ".join(message["content"].split())
            if title:
                return title if len(title) <= max_len else title[: max_len - 1] + "…"
    return "New chat"


def upsert_chat(chats, chat_id: str, messages, limit: int = MAX_HISTORY) -> list[dict]:
    """
    Save a chat to the history (newest first). An existing entry with the same
    id is replaced and moved to the top. Empty chats are not stored.
    """
    others = [c for c in (chats or []) if c.get("id") != chat_id]
    if not messages:
        return others[:limit]
    entry = {"id": chat_id, "title": chat_title(messages), "messages": list(messages)}
    return [entry] + others[: limit - 1]


def find_chat(chats, chat_id: str):
    """Return the chat with this id, or None."""
    return next((c for c in (chats or []) if c.get("id") == chat_id), None)