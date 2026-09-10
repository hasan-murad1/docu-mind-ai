import sys
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

# Allow importing modules from app/core
sys.path.append(str(Path(__file__).parent / "core"))

from document_processor import extract_text, chunk_text
from vector_store import add_chunks_to_store
from rag_pipeline import answer_question

app = FastAPI(title="DocuMind AI")

# --- Security settings ---
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024  # 20 MB limit, prevents huge-file abuse
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"status": "DocuMind AI API is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # 1. Validate file extension (reject anything that isn't pdf/docx)
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. Only PDF and DOCX are allowed.",
        )

    # 2. Generate a safe, unique filename (never trust the user-supplied name directly,
    #    this avoids path traversal attacks like "../../evil.pdf")
    safe_filename = f"{uuid.uuid4().hex}{extension}"
    save_path = UPLOAD_DIR / safe_filename

    # 3. Save the file to disk while enforcing a size limit
    size = 0
    with open(save_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):  # read in 1MB pieces
            size += len(chunk)
            if size > MAX_FILE_SIZE_BYTES:
                buffer.close()
                save_path.unlink(missing_ok=True)  # delete partial file
                raise HTTPException(
                    status_code=400,
                    detail="File too large. Max allowed size is 20MB.",
                )
            buffer.write(chunk)

    # 4. Process the document: extract text, chunk it, store embeddings
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


@app.post("/ask")
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = answer_question(request.question)
    return result