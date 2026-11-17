"""DocuMind AI - FastAPI backend.

/upload : validate, extract, chunk, embed and store a document
/ask    : answer a question with the agentic RAG pipeline (returns a trace)
"""

import sys
import uuid
from pathlib import Path
from typing import Optional

import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

# Allow importing modules from app/core
sys.path.append(str(Path(__file__).parent / "core"))

from agent import DailyCapExceeded, LLMBudgetExceeded, run_agent, validate_question
from document_processor import extract_text, chunk_text
from vector_store import add_chunks_to_store
from usage_limiter import (
    LIMIT_DAY,
    MAX_QUESTIONS_PER_DAY,
    MAX_UPLOADS_PER_DAY,
    question_limiter,
    upload_limiter,
)

app = FastAPI(title="DocuMind AI")

# --- Security settings ---
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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


def enforce_limit(blocked: Optional[str], what: str, per_day: int) -> None:
    """Raise a friendly 429 if a per-visitor limit was hit (blocked is None when allowed)."""
    if blocked is None:
        return
    if blocked == LIMIT_DAY:
        raise HTTPException(
            status_code=429,
            detail=f"You've reached today's limit of {per_day} {what}. Please try again tomorrow.",
        )
    raise HTTPException(
        status_code=429,
        detail=f"Too many {what} in a short time. Please wait a minute and try again.",
        headers={"Retry-After": "60"},
    )


def agent_error_to_http(error: Exception) -> Optional[HTTPException]:
    """Map agent and provider errors to HTTP errors. Unknown errors return None."""
    if isinstance(error, DailyCapExceeded):
        return HTTPException(status_code=429, detail="Daily usage limit reached. Please try again tomorrow.")
    if isinstance(error, LLMBudgetExceeded):
        return HTTPException(
            status_code=429,
            detail="This question needed too many AI steps. Please try a simpler question.",
        )
    if isinstance(error, requests.exceptions.HTTPError):
        if error.response is not None and error.response.status_code == 429:
            return HTTPException(
                status_code=429,
                detail="The AI provider is rate limiting requests. Please try again shortly.",
            )
        return HTTPException(status_code=502, detail="The AI provider returned an error.")
    if isinstance(error, requests.exceptions.RequestException):
        return HTTPException(status_code=503, detail="Could not reach the AI provider.")
    return None


@app.get("/")
def root():
    return {"status": "DocuMind AI API is running"}


async def process_upload(file: UploadFile) -> dict:
    """Validate, save, extract, chunk and store one uploaded file."""
    filename = file.filename or ""

    # 1. Validate file extension
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Only PDF and DOCX are allowed.",
        )

    # 2. Safe, unique filename (prevents path traversal)
    safe_filename = f"{uuid.uuid4().hex}{extension}"
    save_path = UPLOAD_DIR / safe_filename

    # 3. Save file while enforcing the size limit
    size = 0
    with open(save_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                buffer.close()
                save_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=400,
                    detail="File too large. Max allowed size is 20MB.",
                )
            buffer.write(chunk)

    # 4. Extract, chunk, embed, store
    try:
        text = extract_text(str(save_path))
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in file.")

        chunks = chunk_text(text, chunk_size=200, overlap=40)
        stored_count = add_chunks_to_store(chunks, source_filename=filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": filename,
        "chunks_stored": stored_count,
        "message": "Document processed and stored successfully.",
    }


@app.post("/upload")
async def upload_document(http_request: Request, file: UploadFile = File(...)):
    # Per-visitor upload limits (embedding is CPU-heavy)
    visitor = get_client_id(http_request)
    enforce_limit(upload_limiter.try_acquire(visitor), "uploads", MAX_UPLOADS_PER_DAY)

    try:
        return await process_upload(file)
    except HTTPException:
        # Rejected files (wrong type, too large, unreadable) don't use up the quota
        upload_limiter.refund(visitor)
        raise


@app.post("/ask", response_model=AskResponse)
def ask_question(request: QuestionRequest, http_request: Request):
    # 1. Validate the question first (cheap, uses no quota)
    try:
        question = validate_question(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Per-visitor limits: per minute and per day
    visitor = get_client_id(http_request)
    enforce_limit(question_limiter.try_acquire(visitor), "questions", MAX_QUESTIONS_PER_DAY)

    # 3. Run the agent. The daily cap is charged inside the agent for EVERY
    #    LLM call, so there is no upfront daily-cap check here (no double counting).
    try:
        return run_agent(question)
    except Exception as error:
        # The visitor got no answer, so this question doesn't count
        question_limiter.refund(visitor)
        http_error = agent_error_to_http(error)
        if http_error is None:
            raise
        raise http_error from error