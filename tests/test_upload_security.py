"""Tests for upload security: filenames, real file types, size, page and text limits."""

import io
import zipfile

import pytest
from docx import Document
from pypdf import PdfWriter

import upload_security
from upload_security import (
    MAX_EXTRACTED_WORDS,
    MAX_FILENAME_LENGTH,
    MAX_PDF_PAGES,
    MAX_UPLOAD_BYTES,
    UploadRejected,
    check_extracted_text,
    clean_filename,
    detect_file_type,
    validate_upload,
)


def make_pdf(pages=1) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def make_docx(text="Hello from a test document.") -> bytes:
    document = Document()
    document.add_paragraph(text)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_plain_zip() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("notes.txt", "not a word document")
    return buffer.getvalue()


# ---------- Filenames ----------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd.pdf", "passwd.pdf"),
        ("C:\\Users\\x\\My File (1).PDF", "My File (1).pdf"),
        ("re|port*?.pdf", "re_port.pdf"),
        (".hidden.pdf", "hidden.pdf"),
        ("রিপোর্ট.pdf", "রিপোর্ট.pdf"),
        (None, "document"),
        ("", "document"),
    ],
)
def test_clean_filename(raw, expected):
    assert clean_filename(raw) == expected


def test_long_filename_is_shortened_and_keeps_extension():
    name = clean_filename("a" * 300 + ".pdf")
    assert len(name) <= MAX_FILENAME_LENGTH
    assert name.endswith(".pdf")


def test_control_characters_are_removed():
    assert clean_filename("bad\x00name\n.pdf") == "bad_name.pdf"


# ---------- Real file type ----------

def test_detects_pdf():
    assert detect_file_type(make_pdf()) == "pdf"


def test_detects_docx():
    assert detect_file_type(make_docx()) == "docx"


@pytest.mark.parametrize(
    "data",
    [b"MZ\x90\x00 a windows program", b"PK\x03\x04 broken zip", make_plain_zip(), b""],
)
def test_unknown_content_is_not_detected(data):
    assert detect_file_type(data) is None


# ---------- validate_upload ----------

def test_valid_pdf_passes():
    assert validate_upload("report.pdf", make_pdf(3)) == "pdf"


def test_valid_docx_passes():
    assert validate_upload("notes.docx", make_docx()) == "docx"


def test_program_renamed_to_pdf_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("invoice.pdf", b"MZ\x90\x00 a windows program")


def test_pdf_content_with_docx_name_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("notes.docx", make_pdf())


def test_unsupported_extension_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("notes.txt", b"plain text")


def test_empty_file_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("report.pdf", b"")


def test_too_large_file_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("report.pdf", make_pdf() + b"0" * MAX_UPLOAD_BYTES)


def test_page_limit():
    assert validate_upload("ok.pdf", make_pdf(MAX_PDF_PAGES)) == "pdf"
    with pytest.raises(UploadRejected):
        validate_upload("big.pdf", make_pdf(MAX_PDF_PAGES + 1))


def test_damaged_pdf_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("broken.pdf", b"%PDF-1.4 this is not really a pdf")


def test_docx_that_unpacks_too_large_is_rejected(monkeypatch):
    monkeypatch.setattr(upload_security, "MAX_DOCX_UNCOMPRESSED_BYTES", 10)
    with pytest.raises(UploadRejected):
        validate_upload("notes.docx", make_docx())


# ---------- Extracted text ----------

@pytest.mark.parametrize("text", ["", "   \n  ", None])
def test_no_readable_text_is_rejected(text):
    with pytest.raises(UploadRejected):
        check_extracted_text(text)


def test_word_limit():
    assert check_extracted_text("word " * MAX_EXTRACTED_WORDS)
    with pytest.raises(UploadRejected):
        check_extracted_text("word " * (MAX_EXTRACTED_WORDS + 1))


def test_upload_rejected_is_a_value_error():
    assert issubclass(UploadRejected, ValueError)