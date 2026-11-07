"""Retrieve semantically similar PDF chunks from the local AgentX database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.core.settings import ConfigurationError, Settings
from agentx.embeddings import EmbeddingError, OllamaEmbeddingService
from agentx.retrieval import (
    MissingVectorCollectionError,
    SemanticRetrievalError,
    SemanticRetriever,
)
from agentx.storage import ChromaVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="What are the main findings described in the document?",
        help="Natural-language question to search for.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Maximum chunks to show.")
    parser.add_argument("--document-id", help="Optionally limit results to one document.")
    parser.add_argument(
        "--persist-directory",
        help="Chroma directory (defaults to CHROMA_PERSIST_DIRECTORY or ./data/chroma).",
    )
    args = parser.parse_args()

    store: ChromaVectorStore | None = None
    try:
        settings = Settings.from_env()
        store = ChromaVectorStore(args.persist_directory)
        retriever = SemanticRetriever(
            embedding_service=OllamaEmbeddingService(settings=settings),
            vector_store=store,
            top_k=args.top_k,
        )
        results = retriever.retrieve(args.question, document_id=args.document_id)
        print(f"Question: {args.question}")
        print(f"Embedding model: {settings.ollama_embedding_model}")
        if not results:
            print(
                "No matching PDF chunks were found. Store a PDF in this ChromaDB "
                "database first, or try a broader question."
            )
            return 0

        print(f"Retrieved {len(results)} relevant chunk(s):")
        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. {result.filename}, page {result.page_number}"
                f"\n   Chunk: {result.chunk_id}"
                f"\n   Distance: {result.distance:.6f}"
                f"\n   Cosine similarity: {result.similarity:.6f}"
                f"\n   Text: {result.text[:360]}"
            )
        return 0
    except MissingVectorCollectionError as exc:
        print(f"{exc}\nDatabase: {args.persist_directory or './data/chroma'}")
        return 0
    except (ConfigurationError, EmbeddingError, SemanticRetrievalError, ValueError) as exc:
        print(f"Semantic retrieval failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
