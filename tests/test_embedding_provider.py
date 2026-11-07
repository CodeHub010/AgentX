"""Provider selection tests without contacting either embedding provider."""

import unittest
from unittest.mock import Mock, patch

from agentx.core.settings import Settings
from agentx.embeddings import create_embedding_service


class EmbeddingProviderSelectionTests(unittest.TestCase):
    def test_ollama_remains_the_default(self):
        service = Mock(provider="ollama", model_name="nomic-embed-text")
        with patch("agentx.embeddings.ollama_embeddings.OllamaEmbeddingService", return_value=service) as factory:
            selected = create_embedding_service(Settings())
        self.assertIs(selected, service)
        factory.assert_called_once()

    def test_openai_uses_configured_model_when_selected(self):
        settings = Settings(
            embedding_provider="openai",
            openai_embedding_model="text-embedding-test",
            openai_api_key="test-only-key",
        )
        service = Mock(provider="openai", model_name="text-embedding-test")
        with patch("agentx.embeddings.openai_embeddings.OpenAIEmbeddingService", return_value=service) as factory:
            selected = create_embedding_service(settings)
        self.assertIs(selected, service)
        factory.assert_called_once_with(settings=settings)


if __name__ == "__main__":
    unittest.main()
