"""Usage limits: per-client rate limit, global daily LLM cap, per-visitor quotas.

All counters live in memory, so they reset when the server restarts.
"""

import time
from collections import defaultdict, deque
from datetime import date
from typing import Callable, Optional

# --- Settings (tune as needed) ---
MAX_REQUESTS_PER_MINUTE = 10   # per client (generic rate limit)
DAILY_LLM_CALL_CAP = 200       # total calls per day, across all clients
MAX_QUESTION_LENGTH = 500      # characters

# --- Per-visitor limits for the public demo ---
# Only a daily total per visitor. The per-minute value equals the daily one,
# so it never blocks on its own; Groq bursts are handled by the provider retry.
MAX_QUESTIONS_PER_DAY = 5
MAX_UPLOADS_PER_DAY = 3
MAX_QUESTIONS_PER_MINUTE = MAX_QUESTIONS_PER_DAY
MAX_UPLOADS_PER_MINUTE = MAX_UPLOADS_PER_DAY
MAX_TRACKED_VISITORS = 5000    # memory cap for the per-visitor counters
MAX_VISITOR_ID_LENGTH = 200

# Reasons returned by VisitorLimiter.try_acquire when a request is blocked
LIMIT_MINUTE = "minute"
LIMIT_DAY = "day"

# In-memory tracking (resets when the server restarts)
_request_times = defaultdict(deque)
_daily_counter = {"date": date.today(), "count": 0}


def check_rate_limit(client_id: str) -> bool:
    """Return True if this client is within its per-minute limit, else False."""
    now = time.time()
    window_start = now - 60

    # Occasionally remove stale clients so memory can't grow forever
    if len(_request_times) > 1000:
        stale = [cid for cid, ts in _request_times.items() if not ts or ts[-1] < window_start]
        for cid in stale:
            del _request_times[cid]

    timestamps = _request_times[client_id]

    # Drop timestamps older than 60 seconds
    while timestamps and timestamps[0] < window_start:
        timestamps.popleft()

    if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
        return False

    timestamps.append(now)
    return True


def check_and_increment_daily_cap() -> bool:
    """Return True if today's cap isn't reached yet, and count this call."""
    today = date.today()
    if _daily_counter["date"] != today:
        _daily_counter["date"] = today
        _daily_counter["count"] = 0

    if _daily_counter["count"] >= DAILY_LLM_CALL_CAP:
        return False

    _daily_counter["count"] += 1
    return True


# ---------- Per-visitor quotas ----------

def _clean_visitor_id(visitor_id) -> str:
    """Return a safe, bounded visitor id, or raise ValueError."""
    if not isinstance(visitor_id, str) or not visitor_id.strip():
        raise ValueError("visitor_id must be a non-empty string.")
    return visitor_id.strip()[:MAX_VISITOR_ID_LENGTH]


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class VisitorLimiter:
    """
    Per-visitor limits: at most `per_minute` actions in any 60-second window
    and at most `per_day` actions per calendar day.

    A blocked attempt is not recorded. `refund` gives an action back when the
    request failed on our side, so the visitor doesn't lose quota for it.
    `clock` and `today` can be replaced in tests.
    """

    def __init__(
        self,
        per_minute: int,
        per_day: int,
        max_visitors: int = MAX_TRACKED_VISITORS,
        clock: Callable[[], float] = time.time,
        today: Callable[[], date] = date.today,
    ):
        self.per_minute = _positive_int("per_minute", per_minute)
        self.per_day = _positive_int("per_day", per_day)
        self.max_visitors = _positive_int("max_visitors", max_visitors)
        self._clock = clock
        self._today = today
        self._visitors: dict[str, dict] = {}

    def try_acquire(self, visitor_id) -> Optional[str]:
        """
        Record one action for this visitor.
        Returns None if allowed, or LIMIT_DAY / LIMIT_MINUTE if blocked.
        """
        visitor_id = _clean_visitor_id(visitor_id)
        now = self._clock()
        today = self._today()

        entry = self._visitors.get(visitor_id)
        if entry is None:
            self._evict_if_full(today)
            entry = {"day": today, "count": 0, "times": []}
            self._visitors[visitor_id] = entry

        if entry["day"] != today:
            entry["day"] = today
            entry["count"] = 0

        # Keep only actions from the last 60 seconds
        entry["times"] = [t for t in entry["times"] if t > now - 60]

        if entry["count"] >= self.per_day:
            return LIMIT_DAY
        if len(entry["times"]) >= self.per_minute:
            return LIMIT_MINUTE

        entry["count"] += 1
        entry["times"].append(now)
        return None

    def refund(self, visitor_id) -> None:
        """Give back the visitor's most recent action (e.g. the request failed)."""
        entry = self._visitors.get(_clean_visitor_id(visitor_id))
        if entry is None or entry["day"] != self._today():
            return
        if entry["count"] > 0:
            entry["count"] -= 1
        if entry["times"]:
            entry["times"].pop()

    def remaining_today(self, visitor_id) -> int:
        """How many actions this visitor has left today."""
        entry = self._visitors.get(_clean_visitor_id(visitor_id))
        if entry is None or entry["day"] != self._today():
            return self.per_day
        return max(0, self.per_day - entry["count"])

    def reset(self) -> None:
        """Forget every visitor (used in tests)."""
        self._visitors.clear()

    def _evict_if_full(self, today: date) -> None:
        """Keep memory bounded: drop old-day visitors first, then the least recent one."""
        if len(self._visitors) < self.max_visitors:
            return
        stale = [vid for vid, entry in self._visitors.items() if entry["day"] != today]
        for vid in stale:
            del self._visitors[vid]
        if len(self._visitors) >= self.max_visitors:
            oldest = min(
                self._visitors,
                key=lambda vid: max(self._visitors[vid]["times"], default=0.0),
            )
            del self._visitors[oldest]


# Shared limiters used by the API (and later by the Streamlit direct mode)
question_limiter = VisitorLimiter(MAX_QUESTIONS_PER_MINUTE, MAX_QUESTIONS_PER_DAY)
upload_limiter = VisitorLimiter(MAX_UPLOADS_PER_MINUTE, MAX_UPLOADS_PER_DAY)