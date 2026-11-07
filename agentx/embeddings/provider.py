"""Provider-neutral embedding service construction."""

from __future__ import annotations

from typing import Protocol, Iterable, Iterator

from agentx.core.settings import Settings
from agentx.ingestion import PDFChunk
from agentx.embeddings.ollama_embeddings import EmbeddedChunk, EmbeddingConfigurationError


class EmbeddingService(Protocol):
    """Common interface used by ingestion and retrieval components."""

    provider: str
    model_name: str

    def embed_chunks(self, chunks: Iterable[PDFChunk]) -> list[EmbeddedChunk]: ...

    def iter_embeddings(self, chunks: Iterable[PDFChunk]) -> Iterator[EmbeddedChunk]: ...


def create_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Create the embedding service selected by ``EMBEDDING_PROVIDER``."""
    configured = settings or Settings.from_env()
    configured.validate()
    if configured.embedding_provider == "ollama":
        from agentx.embeddings.ollama_embeddings import OllamaEmbeddingService

        return OllamaEmbeddingService(settings=configured)
    if configured.embedding_provider == "openai":
        from agentx.embeddings.openai_embeddings import OpenAIEmbeddingService

        return OpenAIEmbeddingService(settings=configured)
    raise EmbeddingConfigurationError(
        f"Unsupported embedding provider {configured.embedding_provider!r}."
    )
