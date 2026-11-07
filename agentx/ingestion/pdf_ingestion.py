"""PDF text extraction and character-based chunking using PyMuPDF."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


class PDFIngestionError(ValueError):
    """Base exception for PDF validation and ingestion failures."""


class UnsupportedPDFError(PDFIngestionError):
    """Raised for an unsupported input file or document format."""


class InvalidPDFError(PDFIngestionError):
    """Raised when a PDF cannot be read or is invalid/encrypted."""


class EmptyPDFError(PDFIngestionError):
    """Raised when a PDF is empty or has no extractable page text."""


@dataclass(frozen=True, slots=True)
class PDFPageText:
    """Extracted text and source metadata for one non-empty PDF page."""

    document_id: str
    filename: str
    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class PDFChunk:
    """A chunk of PDF text with metadata ready for later retrieval indexing."""

    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str

    @property
    def metadata(self) -> dict[str, str | int]:
        """Return the chunk's source metadata as a plain mapping."""
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
        }


class PDFIngestor:
    """Extract PDF page text and split it into overlapping character chunks.

    Pages are chunked independently so every chunk retains an exact 1-based
    source page number. Empty pages are skipped without disturbing numbering.
    """

    def __init__(self, *, chunk_size: int = 800, chunk_overlap: int = 150) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1 character.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, pdf_path: str | Path, *, document_id: str | None = None) -> list[PDFChunk]:
        """Extract and chunk a PDF, assigning an ID if one is not supplied."""
        pages = self.extract_pages(pdf_path, document_id=document_id)
        return self.chunk_pages(pages)

    def extract_pages(
        self, pdf_path: str | Path, *, document_id: str | None = None
    ) -> list[PDFPageText]:
        """Extract non-empty pages from a PDF using PyMuPDF."""
        path = Path(pdf_path)
        if path.suffix.lower() != ".pdf":
            raise UnsupportedPDFError(
                f"Unsupported file type for {path.name!r}; expected a .pdf file."
            )
        if not path.exists():
            raise InvalidPDFError(f"PDF file does not exist: {path}")
        if not path.is_file():
            raise InvalidPDFError(f"PDF path is not a file: {path}")
        try:
            if path.stat().st_size == 0:
                raise EmptyPDFError(f"PDF file is empty: {path.name}")
        except OSError as exc:
            raise InvalidPDFError(
                f"Could not read PDF file {path}: {type(exc).__name__}."
            ) from exc

        try:
            import pymupdf
        except ImportError as exc:
            raise PDFIngestionError(
                "PyMuPDF is required for PDF ingestion; install dependencies from requirements.txt."
            ) from exc

        try:
            document = pymupdf.open(path)
        except Exception as exc:
            raise InvalidPDFError(
                f"Could not open {path.name!r} as a valid PDF ({type(exc).__name__})."
            ) from exc

        current_document_id = document_id or uuid.uuid4().hex
        pages: list[PDFPageText] = []
        try:
            if not document.is_pdf:
                raise UnsupportedPDFError(f"File is not a supported PDF: {path.name}")
            if document.needs_pass:
                raise InvalidPDFError(
                    f"PDF {path.name!r} is password-protected; encrypted PDFs are not supported."
                )
            for page_index, page in enumerate(document, start=1):
                extracted = page.get_text("text") or ""
                text = re.sub(r"\s+", " ", extracted).strip()
                if text:
                    pages.append(
                        PDFPageText(
                            document_id=current_document_id,
                            filename=path.name,
                            page_number=page_index,
                            text=text,
                        )
                    )
        except PDFIngestionError:
            raise
        except Exception as exc:
            raise InvalidPDFError(
                f"Could not extract text from {path.name!r} ({type(exc).__name__})."
            ) from exc
        finally:
            document.close()

        if not pages:
            raise EmptyPDFError(
                f"PDF {path.name!r} has no extractable text on any page."
            )
        return pages

    def chunk_pages(self, pages: Iterable[PDFPageText]) -> list[PDFChunk]:
        """Split pages while preserving document, file, and page metadata."""
        chunks: list[PDFChunk] = []
        for page in pages:
            for chunk_index, text in enumerate(self._split_text(page.text), start=1):
                chunks.append(
                    PDFChunk(
                        document_id=page.document_id,
                        filename=page.filename,
                        page_number=page.page_number,
                        chunk_id=f"{page.document_id}:p{page.page_number}:c{chunk_index:04d}",
                        text=text,
                    )
                )
        return chunks

    def _split_text(self, text: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", text).strip()
        if not normalized:
            return []

        chunks: list[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + self.chunk_size, len(normalized))
            if end < len(normalized):
                whitespace = normalized.rfind(" ", start + 1, end)
                if whitespace > start + self.chunk_size // 2:
                    end = whitespace
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start = max(start + 1, end - self.chunk_overlap)
        return chunks
