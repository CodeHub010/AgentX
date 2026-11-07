"""Mocked unit tests for the hosted OpenAI embedding adapter."""

import unittest
from unittest.mock import Mock, patch

from agentx.core.settings import Settings
from agentx.embeddings import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    MalformedEmbeddingResponseError,
    OpenAIEmbeddingService,
)
from agentx.ingestion import PDFChunk


class OpenAIEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            embedding_provider="openai",
            openai_embedding_model="text-embedding-test",
            openai_api_key="unit-test-key",
            embedding_batch_size=2,
        )

    @staticmethod
    def chunks(count=3):
        return [
            PDFChunk("doc", "paper.pdf", 1, f"chunk-{i}", f"passage {i}")
            for i in range(count)
        ]

    def test_batches_with_configured_model_and_preserves_identity_and_metadata(self):
        client = Mock()
        client.embed_documents.side_effect = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6]]]
        service = OpenAIEmbeddingService(settings=self.settings, client=client)

        results = service.embed_chunks(self.chunks())

        self.assertEqual(client.embed_documents.call_count, 2)
        self.assertEqual(results[0].provider, "openai")
        self.assertEqual(results[0].model, "text-embedding-test")
        self.assertEqual(results[0].metadata["filename"], "paper.pdf")
        self.assertEqual(results[0].embedding, (0.1, 0.2))

    def test_creates_langchain_client_with_configured_credentials_and_model(self):
        with patch("langchain_openai.OpenAIEmbeddings") as factory:
            service = OpenAIEmbeddingService(settings=self.settings)

        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["model"], "text-embedding-test")
        self.assertEqual(factory.call_args.kwargs["api_key"], "unit-test-key")
        self.assertEqual(service.provider, "openai")

    def test_missing_key_and_wrong_provider_are_rejected(self):
        with self.assertRaisesRegex(EmbeddingConfigurationError, "OPENAI_API_KEY"):
            OpenAIEmbeddingService(settings=Settings(embedding_provider="openai"))
        with self.assertRaisesRegex(EmbeddingConfigurationError, "EMBEDDING_PROVIDER=openai"):
            OpenAIEmbeddingService(settings=Settings(openai_api_key="configured"))

    def test_malformed_count_and_dimensions_are_rejected(self):
        client = Mock()
        client.embed_documents.return_value = [[0.1]]
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "different number"):
            OpenAIEmbeddingService(settings=self.settings, client=client).embed_chunks(self.chunks(2))

        client.embed_documents.side_effect = [[[0.1, 0.2], [0.3]], [[0.4, 0.5, 0.6]]]
        service = OpenAIEmbeddingService(settings=self.settings, client=client)
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "inconsistent dimensions"):
            service.embed_chunks(self.chunks())

    def test_provider_failures_are_clear_and_do_not_expose_key(self):
        client = Mock()
        client.embed_documents.side_effect = RuntimeError("private failure detail")
        service = OpenAIEmbeddingService(settings=self.settings, client=client)
        with self.assertRaises(EmbeddingProviderError) as raised:
            service.embed_chunks(self.chunks(1))
        self.assertNotIn("unit-test-key", str(raised.exception))
        self.assertNotIn("private failure detail", str(raised.exception))

        error = type("AuthenticationError", (Exception,), {})("bad credentials")
        client.embed_documents.side_effect = error
        with self.assertRaises(EmbeddingAuthenticationError):
            service.embed_chunks(self.chunks(1))


if __name__ == "__main__":
    unittest.main()
