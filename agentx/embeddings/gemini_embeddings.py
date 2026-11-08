"""Batched Gemini embeddings using the official LangChain Google GenAI adapter."""

from __future__ import annotations

import math
from numbers import Real
from typing import Iterable, Iterator

from agentx.core.settings import ConfigurationError, Settings
from agentx.embeddings.ollama_embeddings import (
    EmbeddedChunk,
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    EmptyEmbeddingTextError,
    MalformedEmbeddingResponseError,
)
from agentx.ingestion import PDFChunk


class GeminiEmbeddingService:
    """Generate Gemini vectors in bounded batches with stable chunk metadata."""

    provider = "gemini"

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
        client=None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        try:
            self.settings.validate()
        except ConfigurationError as exc:
            raise EmbeddingConfigurationError(str(exc)) from exc
        if self.settings.embedding_provider != self.provider:
            raise EmbeddingConfigurationError(
                "GeminiEmbeddingService requires EMBEDDING_PROVIDER=gemini."
            )
        if not self.settings.gemini_api_key:
            raise EmbeddingConfigurationError(
                "GEMINI_API_KEY is required when EMBEDDING_PROVIDER is 'gemini'."
            )
        self.model_name = (model_name or self.settings.gemini_embedding_model).strip()
        self.batch_size = (
            self.settings.embedding_batch_size if batch_size is None else batch_size
        )
        if not self.model_name:
            raise EmbeddingConfigurationError("Gemini embedding model must not be empty.")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise EmbeddingConfigurationError("Embedding batch_size must be at least 1.")
        self._client = client if client is not None else self._create_client()

    def _create_client(self):
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "Gemini embedding support is unavailable; install langchain-google-genai."
            ) from exc
        try:
            return GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                api_key=self.settings.gemini_api_key,
                request_options={"timeout": self.settings.request_timeout_seconds},
            )
        except Exception as exc:
            raise EmbeddingConfigurationError(
                f"Gemini embedding client could not be initialized ({type(exc).__name__}). "
                "Check the configured embedding model."
            ) from None

    def embed_chunks(self, chunks: Iterable[PDFChunk]) -> list[EmbeddedChunk]:
        return list(self.iter_embeddings(chunks))

    def iter_embeddings(self, chunks: Iterable[PDFChunk]) -> Iterator[EmbeddedChunk]:
        expected_dimension: int | None = None
        batch: list[PDFChunk] = []
        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                raise EmptyEmbeddingTextError(
                    f"PDF chunk {chunk.chunk_id!r} has no text to embed."
                )
            batch.append(chunk)
            if len(batch) >= self.batch_size:
                records, expected_dimension = self._embed_batch(batch, expected_dimension)
                yield from records
                batch = []
        if batch:
            records, _expected_dimension = self._embed_batch(batch, expected_dimension)
            yield from records

    def _embed_batch(
        self, chunks: list[PDFChunk], expected_dimension: int | None
    ) -> tuple[list[EmbeddedChunk], int | None]:
        try:
            vectors = self._client.embed_documents([chunk.text for chunk in chunks])
        except Exception as exc:
            if type(exc).__name__ in {
                "GoogleAuthError",
                "GoogleAuthenticationError",
                "GooglePermissionDeniedError",
                "AuthenticationError",
                "PermissionDenied",
            }:
                raise EmbeddingAuthenticationError(
                    "Google rejected the configured Gemini credentials. Check GEMINI_API_KEY and API access."
                ) from None
            raise EmbeddingProviderError(
                f"Gemini embedding request failed ({type(exc).__name__}). "
                "Check provider availability, quota, and model configuration."
            ) from None
        if not isinstance(vectors, (list, tuple)) or len(vectors) != len(chunks):
            raise MalformedEmbeddingResponseError(
                "Gemini returned a different number of vectors than input chunks."
            )
        output: list[EmbeddedChunk] = []
        dimension = expected_dimension
        for chunk, vector in zip(chunks, vectors):
            if not isinstance(vector, (list, tuple)) or not vector:
                raise MalformedEmbeddingResponseError("Gemini returned an empty embedding vector.")
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise MalformedEmbeddingResponseError(
                    "Gemini returned non-numeric or non-finite embedding values."
                )
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise MalformedEmbeddingResponseError(
                    "Gemini returned vectors with inconsistent dimensions."
                )
            output.append(
                EmbeddedChunk(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    embedding=tuple(float(value) for value in vector),
                    provider=self.provider,
                    model=self.model_name,
                )
            )
        return output, dimension
