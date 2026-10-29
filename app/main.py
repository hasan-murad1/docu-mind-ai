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
from usage_limiter import check_rate_limit

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


@app.get("/")
def root():
    return {"status": "DocuMind AI API is running"}


@app.post("/upload")
async def upload_document(http_request: Request, file: UploadFile = File(...)):
    # 0. Rate limit uploads too (embedding is CPU-heavy)
    if not check_rate_limit(f"upload:{get_client_id(http_request)}"):
        raise HTTPException(
            status_code=429,
            detail="Too many uploads. Please wait a minute and try again.",
        )

    # 1. Validate file extension
    extension = Path(file.filename).suffix.lower()
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
        stored_count = add_chunks_to_store(chunks, source_filename=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "filename": file.filename,
        "chunks_stored": stored_count,
        "message": "Document processed and stored successfully.",
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(request: QuestionRequest, http_request: Request):
    # 1. Validate the question first (cheap, no limits used)
    try:
        question = validate_question(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Per-client rate limit
    if not check_rate_limit(get_client_id(http_request)):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please wait a minute and try again.",
        )

    # 3. Run the agent. The daily cap is charged inside the agent for EVERY
    #    LLM call, so there is no upfront daily-cap check here (no double counting).
    try:
        return run_agent(question)
    except DailyCapExceeded:
        raise HTTPException(
            status_code=429,
            detail="Daily usage limit reached. Please try again tomorrow.",
        )
    except LLMBudgetExceeded:
        raise HTTPException(
            status_code=429,
            detail="This question needed too many AI steps. Please try a simpler question.",
        )
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            raise HTTPException(
                status_code=429,
                detail="The AI provider is rate limiting requests. Please try again shortly.",
            )
        raise HTTPException(status_code=502, detail="The AI provider returned an error.")
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=503, detail="Could not reach the AI provider.")