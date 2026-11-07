"""Batched Ollama embeddings for PDF chunks, without vector-store coupling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Iterable, Iterator, Sequence

from agentx.core.settings import Settings
from agentx.ingestion import PDFChunk


class EmbeddingError(RuntimeError):
    """Base error for local embedding generation."""


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when the embedding client is not configured correctly."""


class EmptyEmbeddingTextError(EmbeddingError, ValueError):
    """Raised when a PDF chunk has no text to embed."""


class EmbeddingConnectionError(EmbeddingError, ConnectionError):
    """Raised when the Ollama embedding endpoint cannot be reached."""


class EmbeddingTimeoutError(EmbeddingConnectionError, TimeoutError):
    """Raised when an Ollama embedding request exceeds its timeout."""


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when the requested local embedding model is not installed."""


class EmbeddingAPIError(EmbeddingError):
    """Raised when Ollama rejects an embedding request."""


class MalformedEmbeddingResponseError(EmbeddingError):
    """Raised when Ollama returns missing, invalid, or misaligned vectors."""


@dataclass(frozen=True, slots=True)
class EmbeddedChunk:
    """Embedding vector paired with the source chunk's metadata."""

    document_id: str
    filename: str
    page_number: int
    chunk_id: str
    embedding: tuple[float, ...]

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
        }


class OllamaEmbeddingService:
    """Generate vectors from Ollama in bounded batches.

    ``iter_embeddings`` consumes any iterable of ``PDFChunk`` objects and
    yields each bounded batch as it completes. It does not retain the full PDF
    or all output vectors unless the caller collects them.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_name: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.settings = settings or Settings.from_env()
        self.settings.validate(provider="ollama")
        selected_model = (
            self.settings.ollama_embedding_model
            if model_name is None
            else model_name
        )
        self.model_name = selected_model.strip()
        self.batch_size = (
            self.settings.embedding_batch_size if batch_size is None else batch_size
        )
        if not self.model_name:
            raise EmbeddingConfigurationError("Embedding model name must not be empty.")
        if self.batch_size < 1:
            raise EmbeddingConfigurationError("Embedding batch_size must be at least 1.")
        self.endpoint = f"{self.settings.ollama_base_url.rstrip('/')}/api/embed"

    def embed_chunks(self, chunks: Iterable[PDFChunk]) -> list[EmbeddedChunk]:
        """Return all embeddings in a list; use ``iter_embeddings`` to stream."""
        return list(self.iter_embeddings(chunks))

    def iter_embeddings(
        self, chunks: Iterable[PDFChunk]
    ) -> Iterator[EmbeddedChunk]:
        """Yield one metadata-preserving vector per chunk in bounded batches."""
        httpx = self._load_httpx()
        timeout = httpx.Timeout(
            timeout=self.settings.request_timeout_seconds,
            connect=self.settings.connect_timeout_seconds,
        )
        batch: list[PDFChunk] = []
        vector_dimension: int | None = None
        try:
            with httpx.Client(timeout=timeout) as client:
                for chunk in chunks:
                    if not chunk.text or not chunk.text.strip():
                        raise EmptyEmbeddingTextError(
                            f"PDF chunk {chunk.chunk_id!r} has no text to embed."
                        )
                    batch.append(chunk)
                    if len(batch) >= self.batch_size:
                        results = self._embed_batch(client, httpx, batch)
                        vector_dimension = self._validate_dimension(
                            results, vector_dimension
                        )
                        yield from results
                        batch = []
                if batch:
                    results = self._embed_batch(client, httpx, batch)
                    self._validate_dimension(results, vector_dimension)
                    yield from results
        except httpx.TimeoutException as exc:
            raise EmbeddingTimeoutError(
                f"Ollama embedding request timed out at {self.settings.ollama_base_url}."
            ) from exc
        except httpx.ConnectError as exc:
            raise EmbeddingConnectionError(
                f"Could not connect to Ollama at {self.settings.ollama_base_url}. "
                "Confirm the Ollama service is running."
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingConnectionError(
                f"Ollama embedding request failed ({type(exc).__name__})."
            ) from exc

    @staticmethod
    def _load_httpx():
        try:
            import httpx
        except ImportError as exc:
            raise EmbeddingConfigurationError(
                "HTTPX is required for Ollama embeddings; install requirements.txt."
            ) from exc
        return httpx

    def _embed_batch(self, client, httpx, chunks: Sequence[PDFChunk]) -> list[EmbeddedChunk]:
        payload = {
            "model": self.model_name,
            "input": [chunk.text for chunk in chunks],
            "truncate": True,
            "keep_alive": "5m",
        }
        try:
            response = client.post(self.endpoint, json=payload)
        except httpx.TimeoutException:
            raise
        except httpx.ConnectError:
            raise
        except httpx.RequestError:
            raise
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._error_detail(exc.response)
            lowered = detail.lower()
            if "model" in lowered and ("not found" in lowered or "pull" in lowered):
                raise EmbeddingModelNotFoundError(
                    f"Ollama embedding model {self.model_name!r} was not found. "
                    f"Install it explicitly with: ollama pull {self.model_name}"
                ) from exc
            raise EmbeddingAPIError(
                f"Ollama embedding API returned HTTP {exc.response.status_code}: "
                f"{detail[:240]}"
            ) from exc

        try:
            data = response.json()
        except Exception as exc:
            raise MalformedEmbeddingResponseError(
                "Ollama returned a response that was not valid JSON."
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("embeddings"), list):
            raise MalformedEmbeddingResponseError(
                "Ollama embedding response must contain an 'embeddings' list."
            )
        vectors = data["embeddings"]
        if len(vectors) != len(chunks):
            raise MalformedEmbeddingResponseError(
                f"Ollama returned {len(vectors)} vectors for {len(chunks)} input chunks."
            )

        output: list[EmbeddedChunk] = []
        expected_dimension: int | None = None
        for chunk, vector in zip(chunks, vectors):
            if not isinstance(vector, (list, tuple)) or not vector:
                raise MalformedEmbeddingResponseError(
                    f"Ollama returned an empty or invalid vector for chunk {chunk.chunk_id!r}."
                )
            if any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                for value in vector
            ):
                raise MalformedEmbeddingResponseError(
                    f"Ollama returned non-numeric or non-finite values for chunk {chunk.chunk_id!r}."
                )
            if expected_dimension is None:
                expected_dimension = len(vector)
            elif len(vector) != expected_dimension:
                raise MalformedEmbeddingResponseError(
                    "Ollama returned vectors with inconsistent dimensions in one batch."
                )
            output.append(
                EmbeddedChunk(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    embedding=tuple(float(value) for value in vector),
                )
            )
        return output

    @staticmethod
    def _validate_dimension(
        results: Sequence[EmbeddedChunk], expected_dimension: int | None
    ) -> int | None:
        if not results:
            return expected_dimension
        dimension = len(results[0].embedding)
        if expected_dimension is not None and dimension != expected_dimension:
            raise MalformedEmbeddingResponseError(
                "Ollama returned vectors with inconsistent dimensions between batches."
            )
        return dimension

    @staticmethod
    def _error_detail(response) -> str:
        try:
            body = response.json()
        except Exception:
            body = None
        if isinstance(body, dict) and body.get("error"):
            return str(body["error"])
        return str(getattr(response, "reason_phrase", "request rejected"))
