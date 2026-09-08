from pathlib import Path
from pypdf import PdfReader
from docx import Document


def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file, page by page."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text())
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


# Quick manual test - only runs when this file is executed directly
if __name__ == "__main__":
    test_file = "data/sample_test.docx"
    extracted_text = extract_text(test_file)
    print("--- EXTRACTED TEXT ---")
    print(extracted_text)
    print("--- END ---")