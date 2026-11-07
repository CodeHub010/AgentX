"""Run grounded question answering over chunks in the local Chroma database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentx.agents import RAGAgent
from agentx.core.settings import ConfigurationError, Settings
from agentx.embeddings import EmbeddingError, OllamaEmbeddingService
from agentx.llm import ModelConfigurationError
from agentx.retrieval import SemanticRetriever
from agentx.storage import ChromaVectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="How does the CNN-LSTM architecture work?",
        help="Question to answer using retrieved PDF passages.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of passages to retrieve.")
    parser.add_argument("--document-id", help="Optionally restrict retrieval to one PDF.")
    parser.add_argument(
        "--persist-directory",
        help="ChromaDB directory (defaults to CHROMA_PERSIST_DIRECTORY or ./data/chroma).",
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
        agent = RAGAgent(settings=settings, retriever=retriever, top_k=args.top_k)
        result = agent.answer(args.question, document_id=args.document_id)

        print(f"Question: {result.question}")
        if result.status == "success":
            print(f"\nAnswer:\n{result.answer}")
        elif result.status == "insufficient_evidence":
            print(f"\nInsufficient evidence:\n{result.answer}")
        else:
            if result.error_message and "MissingVectorCollectionError" in result.error_message:
                print(
                    "\nInsufficient evidence: no PDF vector collection is available. "
                    "Ingest and store a PDF first."
                )
            else:
                print(f"\nRAG failed: {result.answer}", file=sys.stderr)
                if result.error_message:
                    print(result.error_message, file=sys.stderr)
        if result.sources:
            print("\nSources:")
            for source in result.sources:
                score = (
                    f"similarity {source.similarity:.4f}"
                    if source.similarity is not None
                    else "similarity unavailable"
                )
                distance = (
                    f", distance {source.distance:.4f}"
                    if source.distance is not None
                    else ""
                )
                print(
                    f"[{source.number}] {source.filename}, page {source.page_number} "
                    f"({score}{distance}; document {source.document_id}; "
                    f"chunk {source.chunk_id})"
                )
        return 1 if result.status == "error" else 0
    except (ConfigurationError, EmbeddingError, ModelConfigurationError, ValueError) as exc:
        print(f"RAG demo failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()


if __name__ == "__main__":
    raise SystemExit(main())
