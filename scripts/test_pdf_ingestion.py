"""Generate a temporary text PDF and run it through AgentX PDF ingestion."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.ingestion import PDFIngestionError, PDFIngestor


def _create_demo_pdf(path: Path) -> None:
    try:
        import pymupdf
    except ImportError as exc:
        raise PDFIngestionError(
            'PyMuPDF is required. Install this dependency with: python -m pip install "pymupdf>=1.24,<2"'
        ) from exc

    document = pymupdf.open()
    first_page = document.new_page()
    sample_text = (
        "AgentX PDF ingestion extracts text page by page and splits it into "
        "overlapping chunks while keeping filename, document ID, page number, "
        "and chunk ID metadata. "
    ) * 22
    first_page.insert_textbox((50, 50, 550, 740), sample_text, fontsize=10)
    document.new_page()  # Empty pages are allowed and skipped by the text pipeline.
    document.save(path)
    document.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pdf",
        nargs="?",
        help="Optional PDF file to ingest. If omitted, a temporary demo PDF is generated.",
    )
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args()

    ingestor = PDFIngestor(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    try:
        if args.pdf:
            chunks = ingestor.ingest(args.pdf)
            _print_chunks(chunks)
        else:
            with tempfile.TemporaryDirectory(prefix="agentx-pdf-demo-") as temp_dir:
                pdf_path = Path(temp_dir) / "agentx_demo.pdf"
                _create_demo_pdf(pdf_path)
                _print_chunks(ingestor.ingest(pdf_path))
    except (PDFIngestionError, ValueError) as exc:
        print(f"PDF ingestion failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _print_chunks(chunks) -> None:
    print(f"Extracted {len(chunks)} chunk(s).")
    for chunk in chunks:
        print(
            f"\nChunk {chunk.chunk_id}\n"
            f"  Document: {chunk.document_id}\n"
            f"  File: {chunk.filename}\n"
            f"  Page: {chunk.page_number}\n"
            f"  Characters: {len(chunk.text)}\n"
            f"  Text: {chunk.text[:240]}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
