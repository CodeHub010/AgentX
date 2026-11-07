"""Semantic retrieval over chunks already stored in ChromaDB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from agentx.core.settings import Settings
from agentx.embeddings import EmbeddingError, EmbeddingService, create_embedding_service
from agentx.ingestion import PDFChunk
from agentx.storage import (
    ChromaVectorStore,
    StoredChunk,
    VectorStoreCollectionNotFoundError,
    VectorStoreError,
)


class SemanticRetrievalError(RuntimeError):
    """Base error for semantic retrieval failures."""


class RetrievalConfigurationError(SemanticRetrievalError, ValueError):
    """Raised for invalid top-k or retrieval configuration."""


class EmptyQuestionError(SemanticRetrievalError, ValueError):
    """Raised when a retrieval question is blank."""


class RetrievalEmbeddingError(SemanticRetrievalError):
    """Raised when Ollama cannot embed the retrieval question."""


class RetrievalStorageError(SemanticRetrievalError):
    """Raised when ChromaDB cannot execute a vector query."""


class MissingVectorCollectionError(RetrievalStorageError):
    """Raised when the persistent PDF vector collection does not exist yet."""


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A semantically matched PDF text chunk and its source metadata."""

    text: str
    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    distance: float | None
    similarity: float | None


class SemanticRetriever:
    """Embed natural-language questions and retrieve similar stored chunks.

    This component only retrieves source text. It does not call an LLM or
    generate an answer.
    """

    DEFAULT_TOP_K = 5

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        persist_directory: str | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._validate_top_k(top_k)
        self.top_k = top_k
        self.embedding_service = embedding_service or create_embedding_service(settings)
        if vector_store is None:
            self.vector_store = ChromaVectorStore(
                persist_directory,
                embedding_provider=self.embedding_service.provider,
                embedding_model=self.embedding_service.model_name,
            )
        else:
            self.vector_store = vector_store
            store_provider = getattr(vector_store, "embedding_provider", None)
            store_model = getattr(vector_store, "embedding_model", None)
            if isinstance(store_provider, str) and isinstance(store_model, str) and (
                store_provider,
                store_model,
            ) != (self.embedding_service.provider, self.embedding_service.model_name):
                raise RetrievalConfigurationError(
                    "Embedding service and vector store configurations do not match."
                )

    def retrieve(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest chunks for ``question`` from the local collection."""
        if not isinstance(question, str) or not question.strip():
            raise EmptyQuestionError("Question must contain non-whitespace text.")
        selected_top_k = self.top_k if top_k is None else top_k
        self._validate_top_k(selected_top_k)
        if document_id is not None and not document_id.strip():
            raise RetrievalConfigurationError(
                "document_id filter must not be empty when provided."
            )

        query_chunk = PDFChunk(
            document_id="__agentx_query__",
            filename="query.txt",
            page_number=0,
            chunk_id="__agentx_query__",
            text=question.strip(),
        )
        try:
            embedded = self.embedding_service.embed_chunks([query_chunk])
        except EmbeddingError as exc:
            raise RetrievalEmbeddingError(
                f"Could not embed the retrieval question: {exc}"
            ) from exc
        if len(embedded) != 1 or not embedded[0].embedding:
            raise RetrievalEmbeddingError(
                "Embedding service returned no vector for the retrieval question."
            )

        try:
            matches: Sequence[StoredChunk] = self.vector_store.query_by_embedding(
                embedded[0].embedding,
                top_k=selected_top_k,
                document_id=document_id,
            )
        except VectorStoreCollectionNotFoundError as exc:
            raise MissingVectorCollectionError(
                "No PDF vector collection exists yet. Ingest and store a PDF first."
            ) from exc
        except VectorStoreError as exc:
            raise RetrievalStorageError(f"ChromaDB retrieval failed: {exc}") from exc

        return [
            RetrievedChunk(
                text=match.text,
                document_id=match.document_id,
                filename=match.filename,
                page_number=match.page_number,
                chunk_id=match.chunk_id,
                distance=match.distance,
                similarity=match.similarity,
            )
            for match in matches
        ]

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise RetrievalConfigurationError("top_k must be a positive integer.")

