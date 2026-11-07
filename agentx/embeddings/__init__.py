"""Provider configurable embedding-generation services."""

from .ollama_embeddings import (
    EmbeddedChunk,
    EmbeddingAPIError,
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingError,
    EmbeddingModelNotFoundError,
    EmbeddingProviderError,
    EmbeddingTimeoutError,
    EmptyEmbeddingTextError,
    MalformedEmbeddingResponseError,
    OllamaEmbeddingService,
)
from .openai_embeddings import OpenAIEmbeddingService
from .provider import EmbeddingService, create_embedding_service

__all__ = [
    "EmbeddedChunk",
    "EmbeddingAPIError",
    "EmbeddingAuthenticationError",
    "EmbeddingConfigurationError",
    "EmbeddingConnectionError",
    "EmbeddingError",
    "EmbeddingModelNotFoundError",
    "EmbeddingProviderError",
    "EmbeddingTimeoutError",
    "EmptyEmbeddingTextError",
    "MalformedEmbeddingResponseError",
    "OllamaEmbeddingService",
    "OpenAIEmbeddingService",
    "EmbeddingService",
    "create_embedding_service",
]
