"""Upload security: size, file type, page and text limits, and safe filenames.

Pure functions with no web framework, so both the API and the Streamlit
app can use them. Every rejection raises UploadRejected (a ValueError)
with a message that is safe to show to the user.
"""

import io
import re
import unicodedata
import zipfile
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

# --- Limits for the public demo ---
MAX_UPLOAD_BYTES = 5 * 1024 * 1024        # 5 MB
MAX_PDF_PAGES = 30
MAX_EXTRACTED_WORDS = 20_000              # DOCX has no pages, so text length is capped too
MAX_DOCX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024  # protects against "zip bombs"
MAX_DOCX_ENTRIES = 2_000
MAX_FILENAME_LENGTH = 100

# Allowed extension -> detected file type
ALLOWED_TYPES = {".pdf": "pdf", ".docx": "docx"}

_SAFE_PUNCTUATION = " ._-()"


class UploadRejected(ValueError):
    """The uploaded file failed a security or size check."""


# ---------- Filenames ----------

def _is_safe_char(ch: str) -> bool:
    """Letters, marks and digits in any language, plus a few punctuation marks."""
    return ch in _SAFE_PUNCTUATION or unicodedata.category(ch)[0] in "LMN"


def clean_filename(filename) -> str:
    """
    Make a user-supplied filename safe to show and store as a source name:
    keep only the last path part, replace unusual characters with "_",
    shorten it, and keep a lowercase extension.
    """
    if not isinstance(filename, str):
        filename = ""
    base = re.split(r"[\\/]", filename)[-1]

    suffix = Path(base).suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ""
    stem = base[: len(base) - len(suffix)] if suffix else base

    cleaned = "".join(ch if _is_safe_char(ch) else "_" for ch in stem)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    cleaned = cleaned[: MAX_FILENAME_LENGTH - len(suffix)].rstrip(" ._")

    return (cleaned or "document") + suffix


# ---------- File type and content ----------

def detect_file_type(data: bytes) -> Optional[str]:
    """Look at the file's content (not its name). Returns "pdf", "docx" or None."""
    if not isinstance(data, (bytes, bytearray)):
        return None
    if data.startswith(b"%PDF-"):
        return "pdf"
    if data.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names = set(archive.namelist())
        except zipfile.BadZipFile:
            return None
        if "word/document.xml" in names and "[Content_Types].xml" in names:
            return "docx"
    return None


def count_pdf_pages(data: bytes) -> int:
    """Number of pages in a PDF. Encrypted or damaged PDFs are rejected."""
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise UploadRejected("Password-protected PDFs aren't supported.")
        return len(reader.pages)
    except UploadRejected:
        raise
    except Exception as error:  # pypdf raises many different error types
        raise UploadRejected("This PDF couldn't be read. It may be damaged.") from error


def check_docx_archive(data: bytes) -> None:
    """Reject DOCX files that unpack into too much data or too many parts."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as error:
        raise UploadRejected("This DOCX file couldn't be read. It may be damaged.") from error
    if len(entries) > MAX_DOCX_ENTRIES:
        raise UploadRejected("This DOCX file has too many parts.")
    if sum(entry.file_size for entry in entries) > MAX_DOCX_UNCOMPRESSED_BYTES:
        raise UploadRejected("This DOCX file is too large once unpacked.")


def validate_upload(filename, data) -> str:
    """
    Run every check on an uploaded file before it is processed.
    Returns the detected type ("pdf" or "docx"), or raises UploadRejected.
    """
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise UploadRejected("The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadRejected("File too large. Max allowed size is 5 MB.")

    extension = Path(clean_filename(filename)).suffix
    expected = ALLOWED_TYPES.get(extension)
    if expected is None:
        raise UploadRejected(
            f"Unsupported file type '{extension or 'none'}'. Only PDF and DOCX are allowed."
        )

    actual = detect_file_type(data)
    if actual != expected:
        raise UploadRejected(f"This file isn't a real {extension} file.")

    if actual == "pdf":
        pages = count_pdf_pages(data)
        if pages > MAX_PDF_PAGES:
            raise UploadRejected(
                f"Too many pages ({pages}). Max allowed is {MAX_PDF_PAGES} pages."
            )
    else:
        check_docx_archive(data)
    return actual


def check_extracted_text(text) -> str:
    """Reject documents with no readable text or with too much text."""
    if not isinstance(text, str) or not text.strip():
        raise UploadRejected(
            "No readable text found in the file. Scanned PDFs (images only) aren't supported."
        )
    words = len(text.split())
    if words > MAX_EXTRACTED_WORDS:
        raise UploadRejected(
            f"The document is too long ({words:,} words). Max allowed is {MAX_EXTRACTED_WORDS:,} words."
        )
    return text