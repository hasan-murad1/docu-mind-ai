"""Shared fakes for service and API tests: no model, no network, no real database."""

import io
from datetime import date
from types import SimpleNamespace

import requests
from pypdf import PdfWriter

from service import DocuMindService
from usage_limiter import (
    MAX_QUESTIONS_PER_DAY,
    MAX_QUESTIONS_PER_MINUTE,
    MAX_UPLOADS_PER_DAY,
    MAX_UPLOADS_PER_MINUTE,
    VisitorLimiter,
)

SESSION = "visitor_test_collection"

TRACE_STEP = {
    "step": "plan", "tool": "llm", "chunks_retrieved": 0, "best_distance": None,
    "retry": 0, "time_ms": 5.0, "llm_calls": 1,
}
GOOD_RESULT = {"answer": "ChromaDB.", "sources": ["a.pdf"], "trace": [TRACE_STEP]}


def make_pdf(pages=1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


VALID_PDF = make_pdf()


def http_error(status):
    response = requests.Response()
    response.status_code = status
    return requests.exceptions.HTTPError(response=response)


class FakeTime:
    """Controllable clock and calendar day for the limiters."""

    def __init__(self):
        self.now = 1000.0
        self.day = date(2026, 11, 17)

    def clock(self):
        return self.now

    def today(self):
        return self.day

    def next_day(self):
        self.day = date.fromordinal(self.day.toordinal() + 1)
        self.now += 24 * 60 * 60


class FakeSessions:
    """Every visitor gets the same fake collection name."""
    demo_collection = "demo_test"

    def __init__(self):
        self.visitors = []
        self.ended = []

    def collection_for(self, visitor_id):
        self.visitors.append(visitor_id)
        return SESSION

    def expires_in(self, visitor_id):
        return 7200 if visitor_id in self.visitors else None

    def end_session(self, visitor_id):
        self.ended.append(visitor_id)
        return visitor_id in self.visitors


class FakeAgent:
    """Records questions and collections; returns `result` or raises `error`."""

    def __init__(self):
        self.calls = []
        self.collections = []
        self.result = GOOD_RESULT
        self.error = None

    def __call__(self, question, collection_name=None):
        self.calls.append(question)
        self.collections.append(collection_name)
        if self.error is not None:
            raise self.error
        return self.result


def make_service(upload_dir):
    """A DocuMindService wired to fakes, plus handles to inspect them."""
    fake_time = FakeTime()
    questions = VisitorLimiter(MAX_QUESTIONS_PER_MINUTE, MAX_QUESTIONS_PER_DAY,
                               clock=fake_time.clock, today=fake_time.today)
    uploads = VisitorLimiter(MAX_UPLOADS_PER_MINUTE, MAX_UPLOADS_PER_DAY,
                             clock=fake_time.clock, today=fake_time.today)
    sessions = FakeSessions()
    agent = FakeAgent()
    stored_in = []

    def store(chunks, source_filename, collection_name):
        stored_in.append((source_filename, collection_name))
        return len(chunks)

    service = DocuMindService(
        question_limiter=questions,
        upload_limiter=uploads,
        sessions=sessions,
        agent_fn=agent,
        extract_fn=lambda path: "some readable text " * 20,
        store_fn=store,
        sources_fn=lambda collection: [name for name, c in stored_in if c == collection],
        upload_dir=upload_dir,
    )
    return SimpleNamespace(
        service=service, time=fake_time, questions=questions, uploads=uploads,
        sessions=sessions, agent=agent, stored_in=stored_in, upload_dir=upload_dir,
    )