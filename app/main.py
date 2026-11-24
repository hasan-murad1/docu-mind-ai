"""DocuMind AI - FastAPI backend.

A thin HTTP layer over service.py, which holds the real upload and ask flows
(limits, per-visitor sessions, upload security, agent, error handling).

/upload : check and store a document in the visitor's own session
/ask    : answer a question with the agentic RAG pipeline (returns a trace)
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

# Allow importing modules from app/core
sys.path.append(str(Path(__file__).parent / "core"))

from service import DocuMindService, ServiceError
from upload_security import MAX_UPLOAD_BYTES

app = FastAPI(title="DocuMind AI")
app_service = DocuMindService()


class QuestionRequest(BaseModel):
    question: str


class TraceStepOut(BaseModel):
    """Public shape of one trace step. Any extra field is dropped from the response."""

    step: str
    tool: str
    chunks_retrieved: int
    best_distance: Optional[float]
    retry: int
    time_ms: float
    llm_calls: int


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    trace: list[TraceStepOut]


def get_client_id(http_request: Request) -> str:
    return http_request.client.host if http_request.client else "unknown"


def to_http_error(error: ServiceError) -> HTTPException:
    headers = {"Retry-After": str(error.retry_after)} if error.retry_after else None
    return HTTPException(status_code=error.status, detail=error.message, headers=headers)


@app.get("/")
def root():
    return {"status": "DocuMind AI API is running"}


async def read_upload(file: UploadFile) -> bytes:
    """Read the upload into memory, stopping as soon as it passes the size limit."""
    data = bytearray()
    while chunk := await file.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=400, detail="File too large. Max allowed size is 5 MB.")
    return bytes(data)


@app.post("/upload")
async def upload_document(http_request: Request, file: UploadFile = File(...)):
    data = await read_upload(file)
    try:
        # Embedding is CPU-heavy: run it off the event loop
        return await run_in_threadpool(
            app_service.upload, get_client_id(http_request), file.filename, data
        )
    except ServiceError as error:
        raise to_http_error(error) from error


@app.post("/ask", response_model=AskResponse)
def ask_question(request: QuestionRequest, http_request: Request):
    try:
        return app_service.ask(get_client_id(http_request), request.question)
    except ServiceError as error:
        raise to_http_error(error) from error