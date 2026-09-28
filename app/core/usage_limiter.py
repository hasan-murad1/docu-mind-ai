import time
from collections import defaultdict, deque
from datetime import date

# --- Settings (tune as needed) ---
MAX_REQUESTS_PER_MINUTE = 10   # per client
DAILY_LLM_CALL_CAP = 200       # total calls per day, across all clients
MAX_QUESTION_LENGTH = 500      # characters

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