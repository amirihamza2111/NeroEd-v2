from io import BytesIO

import docx
import fitz
import pytest

from extractor import chunk_text, extract_text


def make_pdf(pages: list[str]) -> bytes:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def make_docx(paragraphs: list[str]) -> bytes:
    document = docx.Document()
    for p in paragraphs:
        document.add_paragraph(p)
    buf = BytesIO()
    document.save(buf)
    return buf.getvalue()


class TestExtractText:
    def test_txt(self):
        assert extract_text("héllo wörld".encode("utf-8"), "notes.txt") == "héllo wörld"

    def test_extension_is_case_insensitive(self):
        assert extract_text(b"abc", "NOTES.TXT") == "abc"

    def test_txt_invalid_utf8_raises_value_error(self):
        # UnicodeDecodeError subclasses ValueError, which /upload maps to 400.
        with pytest.raises(ValueError):
            extract_text(b"\xff\xfe\xfa", "bad.txt")

    def test_pdf_all_pages(self):
        text = extract_text(make_pdf(["First page", "Second page"]), "doc.pdf")
        assert "First page" in text
        assert "Second page" in text

    def test_docx_paragraphs(self):
        text = extract_text(make_docx(["Alpha", "Beta"]), "doc.docx")
        assert text == "Alpha\nBeta"

    @pytest.mark.parametrize("name", ["image.png", "noextension", "archive.tar.gz"])
    def test_unsupported(self, name):
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(b"data", name)


class TestChunkText:
    def test_empty(self):
        assert chunk_text("") == []

    def test_short_text_single_chunk(self):
        assert chunk_text("abc", chunk_size=10, overlap=2) == ["abc"]

    def test_sizes_and_overlap(self):
        text = "".join(chr(ord("a") + i % 26) for i in range(1200))
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert all(len(c) <= 500 for c in chunks)
        for prev, nxt in zip(chunks, chunks[1:], strict=False):
            assert prev[-50:] == nxt[:50]

    def test_covers_all_text(self):
        text = "x" * 1001
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        rebuilt = chunks[0] + "".join(c[10:] for c in chunks[1:])
        assert rebuilt == text

    def test_no_redundant_tail_chunk(self):
        # 1000 chars with step 450: the chunk at 450 already reaches the end,
        # so a third chunk would only repeat text from its overlap.
        chunks = chunk_text("y" * 1000, chunk_size=500, overlap=50)
        assert [len(c) for c in chunks] == [500, 500, 100]
        chunks = chunk_text("y" * 950, chunk_size=500, overlap=50)
        assert [len(c) for c in chunks] == [500, 500]

    @pytest.mark.parametrize(
        "size,overlap", [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 20)]
    )
    def test_invalid_params(self, size, overlap):
        with pytest.raises(ValueError):
            chunk_text("abc", chunk_size=size, overlap=overlap)
