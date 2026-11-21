"""Per-visitor document sessions (session isolation).

Every visitor gets their own ChromaDB collection, so one visitor's uploads
never show up in another visitor's answers. A new collection starts with a
copy of the shared demo documents. Visitor collections are deleted after
SESSION_TTL_SECONDS. The creation time is part of the collection name, so
cleanup also works after a server restart.
"""

import re
import threading
import time
import uuid
from typing import Callable, Optional

import vector_store

SESSION_TTL_SECONDS = 2 * 60 * 60   # visitor files live at most 2 hours
MAX_SESSIONS = 500                  # max visitor collections kept at the same time
CLEANUP_INTERVAL_SECONDS = 60       # expired sessions are swept at most once a minute
DEMO_COLLECTION = "demo"            # shared demo documents, copied into each session
VISITOR_PREFIX = "visitor_"
MAX_VISITOR_ID_LENGTH = 200


def _clean_visitor_id(visitor_id) -> str:
    """Return a safe, bounded visitor id, or raise ValueError."""
    if not isinstance(visitor_id, str) or not visitor_id.strip():
        raise ValueError("visitor_id must be a non-empty string.")
    return visitor_id.strip()[:MAX_VISITOR_ID_LENGTH]


def _positive_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


class SessionStore:
    """Maps each visitor to their own collection and deletes old ones."""

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        max_sessions: int = MAX_SESSIONS,
        prefix: str = VISITOR_PREFIX,
        demo_collection: str = DEMO_COLLECTION,
    ):
        if not isinstance(prefix, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,20}_", prefix):
            raise ValueError("prefix must be lowercase letters/digits ending with '_'.")
        self.ttl_seconds = _positive_int("ttl_seconds", ttl_seconds)
        self.max_sessions = _positive_int("max_sessions", max_sessions)
        self.prefix = prefix
        self.demo_collection = demo_collection
        self._clock = clock
        self._visitors: dict[str, str] = {}   # visitor id -> collection name
        self._last_cleanup: Optional[float] = None
        self._lock = threading.RLock()        # Streamlit serves visitors from several threads

    # ---------- Names ----------

    def _created_at(self, name) -> Optional[int]:
        """Creation time stored in a session name, or None if it isn't one of ours."""
        if not isinstance(name, str):
            return None
        match = re.fullmatch(re.escape(self.prefix) + r"(\d+)_[0-9a-f]{12}", name)
        return int(match.group(1)) if match else None

    def _session_names(self) -> list[str]:
        return [n for n in vector_store.list_collection_names() if self._created_at(n) is not None]

    def _delete(self, name: str) -> None:
        vector_store.delete_collection(name)
        self._visitors = {v: n for v, n in self._visitors.items() if n != name}

    # ---------- Public API ----------

    def collection_for(self, visitor_id) -> str:
        """Return this visitor's collection, creating a fresh one when needed."""
        visitor_id = _clean_visitor_id(visitor_id)
        with self._lock:
            self.cleanup_expired()
            now = self._clock()
            name = self._visitors.get(visitor_id)
            if (
                name is not None
                and now - self._created_at(name) < self.ttl_seconds
                and name in self._session_names()
            ):
                return name

            self._make_room()
            name = f"{self.prefix}{int(now)}_{uuid.uuid4().hex[:12]}"
            vector_store.get_or_create_collection(name)
            vector_store.copy_chunks(self.demo_collection, name)
            self._visitors[visitor_id] = name
            return name

    def cleanup_expired(self, force: bool = False) -> int:
        """Delete sessions older than the time limit. Returns how many were deleted."""
        with self._lock:
            now = self._clock()
            if (
                not force
                and self._last_cleanup is not None
                and now - self._last_cleanup < CLEANUP_INTERVAL_SECONDS
            ):
                return 0
            self._last_cleanup = now
            expired = [n for n in self._session_names() if now - self._created_at(n) >= self.ttl_seconds]
            for name in expired:
                self._delete(name)
            return len(expired)

    def expires_in(self, visitor_id) -> Optional[int]:
        """Seconds until this visitor's files are deleted, or None if they have no session."""
        with self._lock:
            name = self._visitors.get(_clean_visitor_id(visitor_id))
            if name is None:
                return None
            return max(0, int(self.ttl_seconds - (self._clock() - self._created_at(name))))

    def end_session(self, visitor_id) -> bool:
        """Delete this visitor's files right away. Returns True if there was a session."""
        with self._lock:
            name = self._visitors.get(_clean_visitor_id(visitor_id))
            if name is None:
                return False
            self._delete(name)
            return True

    def _make_room(self) -> None:
        """Keep at most max_sessions - 1 sessions before adding a new one (oldest go first)."""
        names = sorted(self._session_names(), key=self._created_at)
        extra = len(names) - self.max_sessions + 1
        for name in names[: max(0, extra)]:
            self._delete(name)