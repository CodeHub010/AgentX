"""Mocked tests for Gemini chat and embedding provider selection."""

import sys
import types
import unittest
from unittest.mock import Mock, patch

from agentx.core.settings import ConfigurationError, Settings
from agentx.embeddings import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingProviderError,
    GeminiEmbeddingService,
    MalformedEmbeddingResponseError,
    create_embedding_service,
)
from agentx.ingestion import PDFChunk
from agentx.llm.model_router import ModelConfigurationError, create_chat_model


class GeminiChatTests(unittest.TestCase):
    def test_selects_configured_gemini_chat_model(self):
        model_class = Mock()
        settings = Settings(
            llm_provider="gemini", gemini_api_key="test-key", gemini_model="gemini-test"
        )
        with patch("agentx.llm.model_router._load_chat_gemini", return_value=model_class):
            selected = create_chat_model(settings=settings)
        self.assertIs(selected, model_class.return_value)
        model_class.assert_called_once()
        self.assertEqual(model_class.call_args.kwargs["model"], "gemini-test")
        self.assertEqual(model_class.call_args.kwargs["api_key"], "test-key")

    def test_missing_gemini_chat_key_is_actionable(self):
        with self.assertRaisesRegex(ModelConfigurationError, "GEMINI_API_KEY"):
            create_chat_model(settings=Settings(llm_provider="gemini"))

    def test_gemini_settings_are_read_from_environment_mapping(self):
        settings = Settings.from_env({
            "LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "environment-key",
            "GEMINI_MODEL": "configured-chat",
            "EMBEDDING_PROVIDER": "gemini",
        })
        self.assertEqual(settings.gemini_api_key, "environment-key")
        self.assertEqual(settings.gemini_model, "configured-chat")
        self.assertEqual(settings.gemini_embedding_model, "gemini-embedding-001")
        self.assertNotIn("environment-key", repr(settings))

    def test_invalid_chat_provider_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "choose 'ollama', 'openai', or 'gemini'"):
            Settings.from_env({"LLM_PROVIDER": "not-a-provider"})

    def test_invalid_embedding_provider_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "Unsupported EMBEDDING_PROVIDER"):
            Settings.from_env({"EMBEDDING_PROVIDER": "not-a-provider"})


class GeminiEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            embedding_provider="gemini",
            gemini_api_key="test-key",
            gemini_embedding_model="gemini-embedding-test",
            embedding_batch_size=2,
        )

    @staticmethod
    def chunks(count=3):
        return [
            PDFChunk("doc", "paper.pdf", 1, f"chunk-{index}", f"passage {index}")
            for index in range(count)
        ]

    def test_provider_factory_selects_gemini(self):
        service = Mock(provider="gemini", model_name="gemini-embedding-test")
        with patch(
            "agentx.embeddings.gemini_embeddings.GeminiEmbeddingService",
            return_value=service,
        ) as factory:
            selected = create_embedding_service(self.settings)
        self.assertIs(selected, service)
        factory.assert_called_once_with(settings=self.settings)

    def test_missing_key_and_wrong_provider_are_rejected(self):
        with self.assertRaisesRegex(EmbeddingConfigurationError, "GEMINI_API_KEY"):
            GeminiEmbeddingService(settings=Settings(embedding_provider="gemini"))
        with self.assertRaisesRegex(EmbeddingConfigurationError, "EMBEDDING_PROVIDER=gemini"):
            GeminiEmbeddingService(settings=Settings(gemini_api_key="test-key"))

    def test_batches_and_preserves_provider_model_and_chunk_metadata(self):
        client = Mock()
        client.embed_documents.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6]],
        ]
        service = GeminiEmbeddingService(settings=self.settings, client=client)
        records = service.embed_chunks(self.chunks())
        self.assertEqual(client.embed_documents.call_count, 2)
        self.assertEqual(records[0].provider, "gemini")
        self.assertEqual(records[0].model, "gemini-embedding-test")
        self.assertEqual(records[0].metadata["filename"], "paper.pdf")
        self.assertEqual(records[0].embedding, (0.1, 0.2))

    def test_rejects_vector_dimension_mismatch_within_and_across_batches(self):
        client = Mock()
        client.embed_documents.return_value = [[0.1, 0.2], [0.3]]
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "inconsistent dimensions"):
            GeminiEmbeddingService(settings=self.settings, client=client).embed_chunks(self.chunks(2))

        client.embed_documents.side_effect = [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6, 0.7]]]
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "inconsistent dimensions"):
            GeminiEmbeddingService(settings=self.settings, client=client).embed_chunks(self.chunks())

    def test_langchain_embedding_client_gets_explicit_key_and_model(self):
        factory = Mock()
        fake_module = types.SimpleNamespace(GoogleGenerativeAIEmbeddings=factory)
        with patch.dict(sys.modules, {"langchain_google_genai": fake_module}):
            service = GeminiEmbeddingService(settings=self.settings)
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs["model"], "gemini-embedding-test")
        self.assertEqual(factory.call_args.kwargs["api_key"], "test-key")
        self.assertIs(service._client, factory.return_value)

    def test_provider_errors_do_not_expose_credentials(self):
        key = "secret-key-value"
        service = GeminiEmbeddingService(
            settings=Settings(embedding_provider="gemini", gemini_api_key=key),
            client=Mock(embed_documents=Mock(side_effect=RuntimeError(key))),
        )
        with self.assertRaises(EmbeddingProviderError) as caught:
            service.embed_chunks(self.chunks(1))
        self.assertNotIn(key, str(caught.exception))

    def test_authentication_errors_are_actionable_without_raw_details(self):
        class GoogleAuthError(Exception):
            pass

        service = GeminiEmbeddingService(
            settings=self.settings,
            client=Mock(embed_documents=Mock(side_effect=GoogleAuthError("private"))),
        )
        with self.assertRaisesRegex(EmbeddingAuthenticationError, "GEMINI_API_KEY"):
            service.embed_chunks(self.chunks(1))


if __name__ == "__main__":
    unittest.main()
