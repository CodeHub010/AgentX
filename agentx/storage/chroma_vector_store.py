"""Persistent ChromaDB storage for already-embedded PDF chunks."""

from __future__ import annotations

import os
import hashlib
import math
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Iterable

from agentx.embeddings import EmbeddedChunk
from agentx.ingestion import PDFChunk


class VectorStoreError(RuntimeError):
    """Base error for vector storage operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised for invalid storage configuration or records."""


class DuplicateChunkIDError(VectorStoreConfigurationError):
    """Raised when an input batch contains duplicate chunk IDs."""


class VectorStoreDatabaseError(VectorStoreError):
    """Raised when ChromaDB cannot complete an operation."""


class VectorStoreCollectionNotFoundError(VectorStoreError):
    """Raised when a caller queries a collection that has not been created."""


@dataclass(frozen=True, slots=True)
class StoredChunk:
    """A stored chunk and its precomputed embedding."""

    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    text: str
    embedding: tuple[float, ...]
    distance: float | None = None
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"

    @property
    def similarity(self) -> float | None:
        """Return cosine similarity when Chroma supplied a cosine distance."""
        if self.distance is None:
            return None
        return max(-1.0, min(1.0, 1.0 - self.distance))


class ChromaVectorStore:
    """Persist PDF chunks and their existing vectors in a local Chroma DB.

    Feed matching PDFChunk and EmbeddedChunk iterables. Batches are written as
    they are consumed, so the service does not need to materialize a PDF's
    chunks or vectors before storage. Re-ingestion upserts stable chunk IDs.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        *,
        collection_name: str = "agentx_pdf_chunks",
        batch_size: int = 64,
        embedding_provider: str = "ollama",
        embedding_model: str = "nomic-embed-text",
    ) -> None:
        if not collection_name.strip():
            raise VectorStoreConfigurationError("collection_name must not be empty.")
        if batch_size < 1:
            raise VectorStoreConfigurationError("batch_size must be at least 1.")
        if embedding_provider not in {"ollama", "openai"}:
            raise VectorStoreConfigurationError("Unsupported embedding provider.")
        if not embedding_model or not embedding_model.strip():
            raise VectorStoreConfigurationError("embedding_model must not be empty.")
        configured_path = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "./data/chroma"
        )
        self.persist_directory = Path(configured_path).expanduser()
        self.base_collection_name = collection_name.strip()
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model.strip()
        # Keep the historic default collection (and its local Ollama vectors)
        # intact. Other provider/model pairs get isolated collections.
        if (embedding_provider, self.embedding_model) == ("ollama", "nomic-embed-text"):
            self.collection_name = self.base_collection_name
        else:
            identity = f"{embedding_provider}:{self.embedding_model}"
            suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            self.collection_name = f"{self.base_collection_name[:48]}_{suffix}"
        self.batch_size = batch_size
        self._client = None
        self._collection = None
        self._closed = False

    def __enter__(self) -> "ChromaVectorStore":
        if self._closed:
            raise VectorStoreConfigurationError("This vector store has been closed.")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the collection and close the owned client when supported.

        Recent Chroma clients expose the public ``close()`` API. For older
        versions without it, dropping our references is the only supported
        action; callers that must delete the database immediately should use
        process isolation with those legacy versions.
        """
        if self._closed:
            return
        self._closed = True
        self._collection = None
        client, self._client = self._client, None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def _get_collection(self, *, create: bool = True):
        if self._closed:
            raise VectorStoreConfigurationError("This vector store has been closed.")
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
        except ImportError as exc:
            raise VectorStoreConfigurationError(
                "ChromaDB is required for vector storage; install requirements.txt."
            ) from exc
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(self.persist_directory))
            self._client = client
            if create:
                self._collection = client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                try:
                    self._collection = client.get_collection(name=self.collection_name)
                except Exception as exc:
                    if type(exc).__name__ in {
                        "NotFoundError",
                        "InvalidCollectionException",
                    }:
                        raise VectorStoreCollectionNotFoundError(
                            f"ChromaDB collection {self.collection_name!r} does not exist."
                        ) from exc
                    raise
            return self._collection
        except VectorStoreCollectionNotFoundError:
            raise
        except Exception as exc:
            self.close()
            raise VectorStoreDatabaseError(
                f"Could not open ChromaDB at {self.persist_directory}: {exc}"
            ) from exc

    def add_chunks(
        self,
        chunks: Iterable[PDFChunk],
        embeddings: Iterable[EmbeddedChunk],
    ) -> int:
        """Upsert aligned chunks and precomputed vectors; return records written."""
        sentinel = object()
        chunk_iterator = iter(chunks)
        embedding_iterator = iter(embeddings)
        seen_ids: set[str] = set()
        batch_chunks: list[PDFChunk] = []
        batch_embeddings: list[EmbeddedChunk] = []
        written = 0
        expected_dimension: int | None = None

        def flush() -> int:
            if not batch_chunks:
                return 0
            try:
                self._get_collection().upsert(
                    ids=[chunk.chunk_id for chunk in batch_chunks],
                    documents=[chunk.text for chunk in batch_chunks],
                    embeddings=[list(item.embedding) for item in batch_embeddings],
                    metadatas=[
                        {
                            **chunk.metadata,
                            "embedding_provider": self.embedding_provider,
                            "embedding_model": self.embedding_model,
                        }
                        for chunk in batch_chunks
                    ],
                )
            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreDatabaseError(
                    f"ChromaDB could not store a batch: {exc}"
                ) from exc
            count = len(batch_chunks)
            batch_chunks.clear()
            batch_embeddings.clear()
            return count

        for chunk, embedded in zip_longest(
            chunk_iterator, embedding_iterator, fillvalue=sentinel
        ):
            if chunk is sentinel or embedded is sentinel:
                raise VectorStoreConfigurationError(
                    "Chunk and embedding inputs must contain the same number of records."
                )
            self._validate_pair(chunk, embedded)
            dimension = len(embedded.embedding)
            if expected_dimension is None:
                expected_dimension = dimension
            elif dimension != expected_dimension:
                raise VectorStoreConfigurationError(
                    "Embedding vectors in one ingestion must have consistent dimensions."
                )
            if (embedded.provider, embedded.model) != (
                self.embedding_provider,
                self.embedding_model,
            ):
                raise VectorStoreConfigurationError(
                    f"Embedding identity mismatch for {chunk.chunk_id!r}: "
                    f"received {embedded.provider}/{embedded.model}, expected "
                    f"{self.embedding_provider}/{self.embedding_model}."
                )
            if chunk.chunk_id in seen_ids:
                raise DuplicateChunkIDError(
                    f"Duplicate chunk ID in input: {chunk.chunk_id!r}."
                )
            seen_ids.add(chunk.chunk_id)
            batch_chunks.append(chunk)
            batch_embeddings.append(embedded)
            if len(batch_chunks) >= self.batch_size:
                written += flush()
        written += flush()
        return written

    def get_by_document_id(self, document_id: str) -> list[StoredChunk]:
        """Return all persisted chunks for one document ID."""
        if not document_id or not document_id.strip():
            raise VectorStoreConfigurationError("document_id must not be empty.")
        try:
            response = self._get_collection().get(
                where={"document_id": document_id},
                include=["documents", "metadatas", "embeddings"],
            )
            ids = response.get("ids", [])
            documents = response.get("documents") or []
            metadatas = response.get("metadatas") or []
            vectors = response.get("embeddings")
            if vectors is None:
                raise VectorStoreDatabaseError("ChromaDB returned no embeddings.")
            return [
                StoredChunk(
                    document_id=metadata["document_id"],
                    filename=metadata["filename"],
                    page_number=int(metadata["page_number"]),
                    chunk_id=metadata.get("chunk_id", chunk_id),
                    text=document,
                    embedding=tuple(float(value) for value in vector),
                    embedding_provider=metadata.get("embedding_provider", self.embedding_provider),
                    embedding_model=metadata.get("embedding_model", self.embedding_model),
                )
                for chunk_id, document, metadata, vector in zip(
                    ids, documents, metadatas, vectors
                )
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreDatabaseError(
                f"ChromaDB could not retrieve document {document_id!r}: {exc}"
            ) from exc

    def query_by_embedding(
        self,
        embedding: Iterable[float],
        *,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[StoredChunk]:
        """Query the existing collection using a precomputed vector.

        This method never embeds the query. Distances are returned as provided
        by Chroma; ``similarity`` on each result converts cosine distance to
        ``1 - distance`` because AgentX's collection uses cosine space.
        """
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise VectorStoreConfigurationError("top_k must be a positive integer.")
        query_vector = list(embedding)
        if not query_vector:
            raise VectorStoreConfigurationError("embedding vector must not be empty.")
        if document_id is not None and not document_id.strip():
            raise VectorStoreConfigurationError("document_id filter must not be empty.")
        try:
            filters = {"document_id": document_id} if document_id else None
            response = self._get_collection(create=False).query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )
            ids = (response.get("ids") or [[]])[0]
            documents = (response.get("documents") or [[]])[0]
            metadatas = (response.get("metadatas") or [[]])[0]
            distances = (response.get("distances") or [[]])[0]
            if not ids:
                return []
            if not (len(ids) == len(documents) == len(metadatas) == len(distances)):
                raise VectorStoreDatabaseError(
                    "ChromaDB returned misaligned search results."
                )
            return [
                StoredChunk(
                    document_id=metadata["document_id"],
                    filename=metadata["filename"],
                    page_number=int(metadata["page_number"]),
                    chunk_id=metadata.get("chunk_id", chunk_id),
                    text=document,
                    embedding=(),
                    distance=float(distance),
                    embedding_provider=metadata.get("embedding_provider", self.embedding_provider),
                    embedding_model=metadata.get("embedding_model", self.embedding_model),
                )
                for chunk_id, document, metadata, distance in zip(
                    ids, documents, metadatas, distances
                )
            ]
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreDatabaseError(
                f"ChromaDB vector query failed: {exc}"
            ) from exc

    @staticmethod
    def _validate_pair(chunk: PDFChunk, embedded: EmbeddedChunk) -> None:
        if not isinstance(chunk, PDFChunk) or not isinstance(embedded, EmbeddedChunk):
            raise VectorStoreConfigurationError(
                "add_chunks expects PDFChunk records paired with EmbeddedChunk records."
            )
        if not chunk.chunk_id or not chunk.chunk_id.strip():
            raise VectorStoreConfigurationError("chunk_id must not be empty.")
        if not chunk.text or not chunk.text.strip():
            raise VectorStoreConfigurationError(
                f"Chunk {chunk.chunk_id!r} has empty text."
            )
        if chunk.metadata != embedded.metadata:
            raise VectorStoreConfigurationError(
                f"Chunk and embedding metadata do not match for {chunk.chunk_id!r}."
            )
        if not embedded.embedding:
            raise VectorStoreConfigurationError(
                f"Chunk {chunk.chunk_id!r} has an empty embedding vector."
            )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in embedded.embedding
        ):
            raise VectorStoreConfigurationError(
                f"Chunk {chunk.chunk_id!r} has invalid embedding values."
            )

