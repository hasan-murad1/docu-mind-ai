"""Small helpers for visitor handling in the Streamlit app (no Streamlit import, easy to test)."""

import ipaddress
from typing import Optional

SAMPLE_QUESTIONS = [
    "Who is the CEO of Velora Travel?",
    "What is VT-SUN-02?",
    "I booked a BDT 50,000 trip and cancel 20 days before departure. How much do I get back?",
]

PRIVACY_NOTE = (
    "Your files are only used to answer your questions in this session. Relevant passages "
    "are sent to Groq to write the answers, and everything is deleted within 2 hours. "
    "Refreshing the page starts a new session. Please don't upload confidential documents."
)


def client_ip_from_headers(headers) -> Optional[str]:
    """The visitor's IP from the X-Forwarded-For header (first entry), or None."""
    if not headers:
        return None
    try:
        raw = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
    except Exception:
        return None
    if not isinstance(raw, str):
        return None
    first = raw.split(",")[0].strip()
    try:
        return str(ipaddress.ip_address(first))
    except ValueError:
        return None


def format_duration(seconds) -> str:
    """'1h 52m', '10m', 'less than a minute', or '' when unknown."""
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
        return ""
    minutes = int(seconds // 60)
    if minutes >= 60:
        return f"{minutes // 60}h {minutes % 60}m"
    if minutes >= 1:
        return f"{minutes}m"
    return "less than a minute"


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def usage_line(usage: dict) -> str:
    """One short line about the visitor's remaining quota and file expiry."""
    parts = [
        f"{_count(usage.get('questions_left', 0), 'question')} left today",
        f"{_count(usage.get('uploads_left', 0), 'upload')} left",
    ]
    expires = format_duration(usage.get("expires_in"))
    if expires:
        parts.append(f"files deleted in {expires}")
    return " · ".join(parts)