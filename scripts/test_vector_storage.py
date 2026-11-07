"""Run PDF ingestion, local embedding generation, and persistent storage."""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.core.settings import ConfigurationError, Settings
from agentx.embeddings import EmbeddingError, OllamaEmbeddingService
from agentx.ingestion import PDFIngestionError, PDFIngestor, PDFPageText
from agentx.storage import VectorStoreError, ChromaVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="Optional PDF path; sample text is used by default.")
    parser.add_argument(
        "--persist-directory",
        help="ChromaDB directory (defaults to CHROMA_PERSIST_DIRECTORY or ./data/chroma).",
    )
    args = parser.parse_args()
    document_id = f"storage-demo-{uuid.uuid4().hex[:10]}"

    try:
        settings = Settings.from_env()
        if args.pdf:
            chunks = PDFIngestor().ingest(args.pdf, document_id=document_id)
        else:
            page = PDFPageText(
                document_id=document_id,
                filename="storage-demo.pdf",
                page_number=1,
                text=(
                    "AgentX stores existing local embeddings with searchable PDF metadata. "
                    "This sample is embedded once, then persisted to ChromaDB. "
                ) * 8,
            )
            chunks = PDFIngestor().chunk_pages([page])

        embedding_service = OllamaEmbeddingService(settings=settings)
        store = ChromaVectorStore(args.persist_directory)
        embedded = embedding_service.iter_embeddings(chunks)
        count = store.add_chunks(chunks, embedded)
        records = store.get_by_document_id(document_id)
        print(f"Embedding model: {embedding_service.model_name}")
        print(f"ChromaDB directory: {store.persist_directory.resolve()}")
        print(f"Document ID: {document_id}")
        print(f"Stored {count} chunk vector(s); retrieved {len(records)} record(s).")
        for record in records:
            print(
                f"- {record.chunk_id} | {record.filename}, page {record.page_number} | "
                f"{len(record.embedding)} dimensions"
            )
        return 0
    except (ConfigurationError, EmbeddingError, PDFIngestionError, VectorStoreError, ValueError) as exc:
        print(f"Vector storage demo failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
