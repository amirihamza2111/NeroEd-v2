from io import BytesIO

import docx
import fitz


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract full text from supported file types by extension."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "".join(page.get_text() for page in doc)

    if ext == "docx":
        document = docx.Document(BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if ext == "txt":
        return file_bytes.decode("utf-8")

    raise ValueError(f"Unsupported file type: {ext}")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0

    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        if start + chunk_size >= len(text):
            # This chunk reached the end; another would only repeat its tail.
            break
        start += step

    return chunks
