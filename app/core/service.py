"""Application service: the full upload and ask flows in one place.

Both front doors use it: the FastAPI backend (app/main.py) and the Streamlit
app in direct mode. So per-visitor limits, session isolation, upload security
and error handling behave the same everywhere. Problems are raised as
ServiceError with a status code and a message that is safe to show to users.

Two ids per visitor:
- visitor_id picks the visitor's own document collection (one per browser session)
- quota_id counts their daily questions and uploads (e.g. their IP address);
  it defaults to visitor_id
"""

import uuid
from pathlib import Path
from typing import Callable, Optional

import requests

import vector_store
from agent import DailyCapExceeded, LLMBudgetExceeded, run_agent, validate_question
from document_processor import chunk_text, extract_text
from sessions import SessionStore
from upload_security import UploadRejected, check_extracted_text, clean_filename, validate_upload
from usage_limiter import LIMIT_DAY
from usage_limiter import question_limiter as shared_question_limiter
from usage_limiter import upload_limiter as shared_upload_limiter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_DIR = Path("data/uploads")
DEMO_DOCS_DIR = PROJECT_ROOT / "demo_docs"   # shared demo documents committed to the repo
MAX_DEMO_FILES = 10
DEMO_EXTENSIONS = {".pdf", ".docx"}
CHUNK_SIZE = 200
CHUNK_OVERLAP = 40


class ServiceError(Exception):
    """A request that could not be completed. `message` is safe to show to users."""

    def __init__(self, status: int, message: str, retry_after: Optional[int] = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


def agent_error_to_service_error(error: Exception) -> Optional[ServiceError]:
    """Map agent and provider errors to user-safe errors. Unknown errors return None."""
    if isinstance(error, DailyCapExceeded):
        return ServiceError(429, "Daily usage limit reached. Please try again tomorrow.")
    if isinstance(error, LLMBudgetExceeded):
        return ServiceError(429, "This question needed too many AI steps. Please try a simpler question.")
    if isinstance(error, requests.exceptions.HTTPError):
        if error.response is not None and error.response.status_code == 429:
            return ServiceError(429, "The AI provider is rate limiting requests. Please try again shortly.")
        return ServiceError(502, "The AI provider returned an error.")
    if isinstance(error, requests.exceptions.RequestException):
        return ServiceError(503, "Could not reach the AI provider.")
    return None


class DocuMindService:
    """Upload and ask flows for one visitor at a time. Every dependency can be replaced in tests."""

    def __init__(
        self,
        question_limiter=None,
        upload_limiter=None,
        sessions=None,
        agent_fn: Optional[Callable] = None,
        extract_fn: Optional[Callable[[str], str]] = None,
        store_fn: Optional[Callable] = None,
        sources_fn: Optional[Callable[[str], list]] = None,
        upload_dir=None,
    ):
        self.question_limiter = question_limiter if question_limiter is not None else shared_question_limiter
        self.upload_limiter = upload_limiter if upload_limiter is not None else shared_upload_limiter
        self.sessions = sessions if sessions is not None else SessionStore()
        self.agent_fn = agent_fn or run_agent
        self.extract_fn = extract_fn or extract_text
        self.store_fn = store_fn or vector_store.add_chunks_to_store
        self.sources_fn = sources_fn or vector_store.list_sources
        self.upload_dir = Path(upload_dir) if upload_dir is not None else DEFAULT_UPLOAD_DIR
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Limits ----------

    @staticmethod
    def _check_limit(blocked: Optional[str], what: str, limiter) -> None:
        """Raise a friendly 429 if a per-visitor limit was hit (blocked is None when allowed)."""
        if blocked is None:
            return
        if blocked == LIMIT_DAY:
            raise ServiceError(
                429, f"You've reached today's limit of {limiter.per_day} {what}. Please try again tomorrow."
            )
        raise ServiceError(
            429, f"Too many {what} in a short time. Please wait a minute and try again.", retry_after=60
        )

    # ---------- Demo documents ----------

    def load_demo_documents(self, demo_dir=None) -> int:
        """
        Load the shared demo documents into the demo collection, once.
        Returns how many files were loaded (0 if they were already there).
        """
        demo_collection = self.sessions.demo_collection
        if self.sources_fn(demo_collection):
            return 0
        folder = Path(demo_dir) if demo_dir is not None else DEMO_DOCS_DIR
        if not folder.is_dir():
            return 0

        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in DEMO_EXTENSIONS)
        for path in files[:MAX_DEMO_FILES]:
            text = check_extracted_text(self.extract_fn(str(path)))
            chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            self.store_fn(chunks, source_filename=path.name, collection_name=demo_collection)
        return len(files[:MAX_DEMO_FILES])

    # ---------- Upload ----------

    def upload(self, visitor_id: str, filename, data, quota_id: Optional[str] = None) -> dict:
        """Check and store one document in this visitor's own collection."""
        quota = quota_id or visitor_id
        self._check_limit(self.upload_limiter.try_acquire(quota), "uploads", self.upload_limiter)
        try:
            collection = self.sessions.collection_for(visitor_id)
            return self._store_document(collection, filename, data)
        except ServiceError:
            # Rejected files (wrong type, too large, unreadable) don't use up the quota
            self.upload_limiter.refund(quota)
            raise

    def _store_document(self, collection: str, filename, data) -> dict:
        display_name = clean_filename(filename)

        # 1. Security checks on the real content (size, type, pages)
        try:
            kind = validate_upload(display_name, data)
        except UploadRejected as error:
            raise ServiceError(400, str(error)) from error

        # 2. Save under a random name only while extracting the text
        save_path = self.upload_dir / f"{uuid.uuid4().hex}.{kind}"
        try:
            save_path.write_bytes(data)
            try:
                raw_text = self.extract_fn(str(save_path))
            except Exception as error:
                raise ServiceError(400, "This file couldn't be read. It may be damaged.") from error

            text = check_extracted_text(raw_text)
            chunks = chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
            stored = self.store_fn(chunks, source_filename=display_name, collection_name=collection)
        except ValueError as error:  # includes UploadRejected
            raise ServiceError(400, str(error)) from error
        finally:
            # Privacy: the original file isn't needed after extraction
            save_path.unlink(missing_ok=True)

        return {
            "filename": display_name,
            "chunks_stored": stored,
            "message": "Document processed and stored successfully.",
        }

    # ---------- Ask ----------

    def ask(self, visitor_id: str, question, quota_id: Optional[str] = None) -> dict:
        """Answer a question from this visitor's own documents. Returns answer, sources and trace."""
        try:
            question = validate_question(question)
        except ValueError as error:
            raise ServiceError(400, str(error)) from error

        quota = quota_id or visitor_id
        self._check_limit(self.question_limiter.try_acquire(quota), "questions", self.question_limiter)
        try:
            collection = self.sessions.collection_for(visitor_id)
            return self.agent_fn(question, collection_name=collection)
        except Exception as error:
            # The visitor got no answer, so this question doesn't count
            self.question_limiter.refund(quota)
            mapped = agent_error_to_service_error(error)
            if mapped is None:
                raise
            raise mapped from error

    # ---------- Session info ----------

    def documents(self, visitor_id: str) -> list[str]:
        """Names of the documents this visitor can ask about."""
        return self.sources_fn(self.sessions.collection_for(visitor_id))

    def usage(self, visitor_id: str, quota_id: Optional[str] = None) -> dict:
        """What this visitor has left today, and when their files will be deleted."""
        quota = quota_id or visitor_id
        return {
            "questions_left": self.question_limiter.remaining_today(quota),
            "uploads_left": self.upload_limiter.remaining_today(quota),
            "expires_in": self.sessions.expires_in(visitor_id),
        }

    def end_session(self, visitor_id: str) -> bool:
        """Delete this visitor's files right away."""
        return self.sessions.end_session(visitor_id)