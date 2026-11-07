"""Local embedding-generation services."""

from .ollama_embeddings import (
    EmbeddedChunk,
    EmbeddingAPIError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingModelNotFoundError,
    EmbeddingTimeoutError,
    EmptyEmbeddingTextError,
    MalformedEmbeddingResponseError,
    OllamaEmbeddingService,
)

__all__ = [
    "EmbeddedChunk",
    "EmbeddingAPIError",
    "EmbeddingConfigurationError",
    "EmbeddingConnectionError",
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmbeddingTimeoutError",
    "EmptyEmbeddingTextError",
    "MalformedEmbeddingResponseError",
    "OllamaEmbeddingService",
]
