"""Demonstrate local Ollama embeddings for PDF chunks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.core.settings import ConfigurationError, Settings
from agentx.embeddings import (
    EmbeddingError,
    EmbeddingModelNotFoundError,
    OllamaEmbeddingService,
)
from agentx.ingestion import PDFIngestionError, PDFIngestor, PDFPageText


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        help="Optional PDF path; if omitted, generated sample PDF chunks are embedded.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override OLLAMA_EMBEDDING_BATCH_SIZE for this run.",
    )
    args = parser.parse_args()

    try:
        settings = Settings.from_env()
        if args.pdf:
            chunks = PDFIngestor().ingest(args.pdf)
        else:
            demo_page = PDFPageText(
                document_id="embedding-demo",
                filename="demo.pdf",
                page_number=1,
                text=(
                    "AgentX creates local embeddings for PDF text chunks. "
                    "Each vector keeps the source document, filename, page, and chunk identifiers. "
                ) * 8,
            )
            chunks = PDFIngestor().chunk_pages([demo_page])

        service = OllamaEmbeddingService(
            settings=settings,
            batch_size=args.batch_size,
        )
        results = service.iter_embeddings(chunks)
        count = 0
        print(f"Provider: Ollama ({settings.ollama_base_url})")
        print(f"Embedding model: {service.model_name}")
        print(f"Batch size: {service.batch_size}")
        for result in results:
            count += 1
            preview = ", ".join(f"{value:.6f}" for value in result.embedding[:8])
            print(
                f"\nChunk {result.chunk_id}\n"
                f"  File: {result.filename}\n"
                f"  Document: {result.document_id}\n"
                f"  Page: {result.page_number}\n"
                f"  Vector dimensions: {len(result.embedding)}\n"
                f"  Vector preview: [{preview}{', ...' if len(result.embedding) > 8 else ''}]"
            )
        print(f"\nGenerated {count} embedding vector(s).")
        if count == 0:
            print("No chunks were supplied.")
        return 0
    except EmbeddingModelNotFoundError as exc:
        print(f"Embedding model not found: {exc}", file=sys.stderr)
        return 1
    except (EmbeddingError, PDFIngestionError, ConfigurationError, ValueError) as exc:
        print(f"Embedding demo failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
