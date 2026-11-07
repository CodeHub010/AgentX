"""Unit tests for PDF validation, extraction, chunking, and metadata."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentx.ingestion import (
    EmptyPDFError,
    InvalidPDFError,
    PDFIngestor,
    UnsupportedPDFError,
)


class FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def get_text(self, mode: str) -> str:
        if mode != "text":
            raise AssertionError(f"Unexpected extraction mode: {mode}")
        return self._text


class FakeDocument:
    is_pdf = True
    needs_pass = False

    def __init__(self, texts: list[str]) -> None:
        self.pages = [FakePage(text) for text in texts]
        self.closed = False

    def __iter__(self):
        return iter(self.pages)

    def close(self) -> None:
        self.closed = True


class PDFIngestorTests(unittest.TestCase):
    def make_pdf_path(self, directory: str, name: str = "source.pdf", data: bytes = b"%PDF-mock") -> Path:
        path = Path(directory) / name
        path.write_bytes(data)
        return path

    def ingest_with_pages(self, path: Path, document: FakeDocument, *, chunk_size: int = 800, overlap: int = 150):
        fake_pymupdf = SimpleNamespace(open=Mock(return_value=document))
        ingestor = PDFIngestor(chunk_size=chunk_size, chunk_overlap=overlap)
        with patch.dict("sys.modules", {"pymupdf": fake_pymupdf}):
            return ingestor.ingest(path, document_id="doc-test"), fake_pymupdf

    def test_extracts_pages_skips_empty_page_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.make_pdf_path(temp_dir, "Original Name.pdf")
            long_text = " ".join(f"token{i}" for i in range(180))
            document = FakeDocument(["First page text.", " \n\t ", long_text])
            chunks, fake_pymupdf = self.ingest_with_pages(
                path, document, chunk_size=100, overlap=20
            )

        self.assertGreater(len(chunks), 2)
        self.assertTrue(document.closed)
        self.assertEqual(fake_pymupdf.open.call_args.args[0].name, "Original Name.pdf")
        self.assertEqual(chunks[0].filename, "Original Name.pdf")
        self.assertEqual(chunks[0].document_id, "doc-test")
        self.assertEqual(chunks[0].page_number, 1)
        self.assertTrue(all(chunk.page_number == 3 for chunk in chunks[1:]))
        self.assertEqual(chunks[0].metadata["chunk_id"], chunks[0].chunk_id)
        self.assertTrue(chunks[0].chunk_id.startswith("doc-test:p1:c"))
        self.assertTrue(all(len(chunk.text) <= 100 for chunk in chunks))

    def test_chunk_overlap_is_preserved(self) -> None:
        ingestor = PDFIngestor(chunk_size=20, chunk_overlap=5)
        chunks = ingestor.chunk_pages(
            [
                SimpleNamespace(
                    document_id="d", filename="f.pdf", page_number=1,
                    text="abcdefghijklmnopqrstuvwxyz" * 3,
                )
            ]
        )
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0].text[-5:], chunks[1].text[:5])

    def test_unsupported_extension_is_rejected_before_parser(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.make_pdf_path(temp_dir, "notes.txt")
            with self.assertRaisesRegex(UnsupportedPDFError, "expected a .pdf"):
                PDFIngestor().ingest(path)

    def test_empty_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.make_pdf_path(temp_dir, data=b"")
            with self.assertRaisesRegex(EmptyPDFError, "file is empty"):
                PDFIngestor().ingest(path)

    def test_invalid_pdf_is_reported_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.make_pdf_path(temp_dir)
            fake_pymupdf = SimpleNamespace(open=Mock(side_effect=RuntimeError("not a PDF")))
            with patch.dict("sys.modules", {"pymupdf": fake_pymupdf}):
                with self.assertRaisesRegex(InvalidPDFError, "valid PDF"):
                    PDFIngestor().ingest(path)

    def test_all_empty_pages_are_rejected_as_empty_content(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.make_pdf_path(temp_dir)
            document = FakeDocument(["", "  \n "])
            with self.assertRaisesRegex(EmptyPDFError, "no extractable text"):
                self.ingest_with_pages(path, document)

    def test_non_pdf_document_opened_by_parser_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self.make_pdf_path(temp_dir)
            document = FakeDocument(["not actually a PDF"])
            document.is_pdf = False
            fake_pymupdf = SimpleNamespace(open=Mock(return_value=document))
            with patch.dict("sys.modules", {"pymupdf": fake_pymupdf}):
                with self.assertRaisesRegex(UnsupportedPDFError, "not a supported PDF"):
                    PDFIngestor().ingest(path)
            self.assertTrue(document.closed)

    def test_chunk_configuration_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "smaller than chunk_size"):
            PDFIngestor(chunk_size=10, chunk_overlap=10)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            PDFIngestor(chunk_size=10, chunk_overlap=-1)


if __name__ == "__main__":
    unittest.main()
