"""Extract text from supported resume formats."""
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_file(filename: str, content: bytes) -> str:
    extension = Path(filename).suffix.lower()
    if extension == ".txt":
        return content.decode("utf-8-sig", errors="replace")
    if extension == ".pdf":
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError("Unsupported file type. Upload a PDF or TXT file.")
