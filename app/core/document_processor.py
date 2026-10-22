"""Text extraction (PDF/DOCX) and word-based chunking."""

from pathlib import Path

from docx import Document
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        # Some pages (e.g. scanned images) have no text layer
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def extract_text_from_docx(file_path: str) -> str:
    """Extract all text from a DOCX file, paragraph by paragraph."""
    doc = Document(file_path)
    text_parts = [para.text for para in doc.paragraphs]
    return "\n".join(text_parts)


def extract_text(file_path: str) -> str:
    """Detect file type from extension and call the right extractor."""
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_text_from_pdf(file_path)
    elif extension == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def _is_int(value) -> bool:
    """True for real integers only (bool is rejected even though it is an int subclass)."""
    return isinstance(value, int) and not isinstance(value, bool)


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks of roughly `chunk_size` words.

    Raises ValueError for invalid settings. `overlap` must be smaller than
    `chunk_size`, otherwise the window would never move forward.
    """
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if not _is_int(chunk_size) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if not _is_int(overlap) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    words = text.split()
    step = chunk_size - overlap  # always >= 1 after validation

    # A bounded for-loop: it always finishes, no while-loop needed
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
    ]


if __name__ == "__main__":
    test_file = "data/sample_test.docx"
    extracted_text = extract_text(test_file)

    chunks = chunk_text(extracted_text, chunk_size=30, overlap=5)
    print(f"Total chunks created: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)