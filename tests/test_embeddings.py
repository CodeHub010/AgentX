"""Mocked HTTP tests for batched Ollama embedding generation."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentx.core.settings import ConfigurationError, Settings
from agentx.embeddings import (
    EmbeddingConfigurationError,
    EmbeddingConnectionError,
    EmbeddingModelNotFoundError,
    EmbeddingTimeoutError,
    EmptyEmbeddingTextError,
    MalformedEmbeddingResponseError,
    OllamaEmbeddingService,
)
from agentx.ingestion import PDFChunk


class FakeRequestError(Exception):
    pass


class FakeConnectError(FakeRequestError):
    pass


class FakeTimeoutException(FakeRequestError):
    pass


class FakeHTTPStatusError(Exception):
    def __init__(self, response) -> None:
        self.response = response
        super().__init__("HTTP request failed")


class FakeResponse:
    def __init__(self, body: dict, status_code: int = 200) -> None:
        self._body = body
        self.status_code = status_code
        self.reason_phrase = "Request rejected"

    def json(self):
        return self._body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise FakeHTTPStatusError(self)


class EmbeddingServiceTests(unittest.TestCase):
    def make_chunk(self, number: int, text: str | None = None) -> PDFChunk:
        return PDFChunk(
            document_id="doc-1",
            filename="source.pdf",
            page_number=number,
            chunk_id=f"doc-1:p{number}:c0001",
            text=text if text is not None else f"Text for page {number}",
        )

    def fake_httpx(self, responses: list[FakeResponse] | None = None, error=None):
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        if error is not None:
            client.post.side_effect = error
        else:
            client.post.side_effect = responses or []
        module = SimpleNamespace(
            Timeout=lambda **values: SimpleNamespace(**values),
            Client=Mock(return_value=client),
            TimeoutException=FakeTimeoutException,
            ConnectError=FakeConnectError,
            RequestError=FakeRequestError,
            HTTPStatusError=FakeHTTPStatusError,
        )
        return module, client

    def run_with_httpx(self, service, chunks, fake_httpx):
        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            return service.embed_chunks(chunks)

    def test_default_model_and_batch_settings(self) -> None:
        settings = Settings.from_env({})
        self.assertEqual(settings.ollama_embedding_model, "nomic-embed-text")
        self.assertGreaterEqual(settings.embedding_batch_size, 1)

    def test_batches_inputs_and_preserves_chunk_metadata(self) -> None:
        settings = Settings(embedding_batch_size=2)
        service = OllamaEmbeddingService(settings=settings)
        responses = [
            FakeResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]}),
            FakeResponse({"embeddings": [[0.5, 0.6]]}),
        ]
        fake_httpx, client = self.fake_httpx(responses)
        chunks = (self.make_chunk(index) for index in range(1, 4))

        embedded = self.run_with_httpx(service, chunks, fake_httpx)

        self.assertEqual(len(embedded), 3)
        self.assertEqual(embedded[0].embedding, (0.1, 0.2))
        self.assertEqual(embedded[1].metadata["filename"], "source.pdf")
        self.assertEqual(embedded[2].metadata["page_number"], 3)
        self.assertEqual(embedded[2].chunk_id, "doc-1:p3:c0001")
        self.assertEqual(client.post.call_count, 2)
        self.assertEqual(
            client.post.call_args_list[0].kwargs["json"]["input"],
            ["Text for page 1", "Text for page 2"],
        )
        self.assertEqual(
            client.post.call_args_list[0].args[0],
            "http://localhost:11434/api/embed",
        )
        self.assertEqual(
            client.post.call_args_list[0].kwargs["json"]["model"],
            "nomic-embed-text",
        )

    def test_iter_embeddings_yields_as_each_bounded_batch_completes(self) -> None:
        service = OllamaEmbeddingService(settings=Settings(embedding_batch_size=1))
        fake_httpx, client = self.fake_httpx(
            [FakeResponse({"embeddings": [[0.1]]}), FakeResponse({"embeddings": [[0.2]]})]
        )
        with patch.dict(sys.modules, {"httpx": fake_httpx}):
            iterator = service.iter_embeddings(self.make_chunk(i) for i in (1, 2))
            first = next(iterator)
            self.assertEqual(first.embedding, (0.1,))
            self.assertEqual(client.post.call_count, 1)
            second = next(iterator)
            self.assertEqual(second.embedding, (0.2,))
            self.assertEqual(client.post.call_count, 2)

    def test_empty_text_is_rejected_without_request(self) -> None:
        service = OllamaEmbeddingService()
        fake_httpx, client = self.fake_httpx()
        with self.assertRaisesRegex(EmptyEmbeddingTextError, "has no text"):
            self.run_with_httpx(service, [self.make_chunk(1, "  ")], fake_httpx)
        client.post.assert_not_called()

    def test_missing_or_misaligned_vectors_are_rejected(self) -> None:
        service = OllamaEmbeddingService(settings=Settings(embedding_batch_size=2))
        fake_httpx, _ = self.fake_httpx([FakeResponse({"embeddings": [[0.1]]})])
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "1 vectors for 2"):
            self.run_with_httpx(
                service, [self.make_chunk(1), self.make_chunk(2)], fake_httpx
            )

    def test_non_numeric_or_inconsistent_vectors_are_rejected(self) -> None:
        service = OllamaEmbeddingService(settings=Settings(embedding_batch_size=2))
        fake_httpx, _ = self.fake_httpx(
            [FakeResponse({"embeddings": [[0.1], [float("nan")]]})]
        )
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "non-numeric"):
            self.run_with_httpx(
                service, [self.make_chunk(1), self.make_chunk(2)], fake_httpx
            )

        fake_httpx, _ = self.fake_httpx(
            [FakeResponse({"embeddings": [[0.1], [0.2, 0.3]]})]
        )
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "inconsistent dimensions"):
            self.run_with_httpx(
                service, [self.make_chunk(1), self.make_chunk(2)], fake_httpx
            )

    def test_inconsistent_dimensions_across_batches_are_rejected(self) -> None:
        service = OllamaEmbeddingService(settings=Settings(embedding_batch_size=1))
        fake_httpx, _ = self.fake_httpx(
            [FakeResponse({"embeddings": [[0.1]]}), FakeResponse({"embeddings": [[0.2, 0.3]]})]
        )
        with self.assertRaisesRegex(MalformedEmbeddingResponseError, "between batches"):
            self.run_with_httpx(
                service, [self.make_chunk(1), self.make_chunk(2)], fake_httpx
            )

    def test_model_not_found_has_explicit_manual_install_guidance(self) -> None:
        service = OllamaEmbeddingService()
        fake_httpx, _ = self.fake_httpx(
            [FakeResponse({"error": "model 'nomic-embed-text' not found"}, status_code=404)]
        )
        with self.assertRaisesRegex(EmbeddingModelNotFoundError, "ollama pull nomic-embed-text"):
            self.run_with_httpx(service, [self.make_chunk(1)], fake_httpx)

    def test_connection_and_timeout_errors_are_distinct(self) -> None:
        service = OllamaEmbeddingService()
        fake_httpx, _ = self.fake_httpx(error=FakeConnectError("offline"))
        with self.assertRaisesRegex(EmbeddingConnectionError, "Could not connect"):
            self.run_with_httpx(service, [self.make_chunk(1)], fake_httpx)

        fake_httpx, _ = self.fake_httpx(error=FakeTimeoutException("slow"))
        with self.assertRaisesRegex(EmbeddingTimeoutError, "timed out"):
            self.run_with_httpx(service, [self.make_chunk(1)], fake_httpx)

    def test_batch_size_and_model_configuration_are_validated(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "positive integer"):
            Settings.from_env({"OLLAMA_EMBEDDING_BATCH_SIZE": "many"})
        with self.assertRaisesRegex(EmbeddingConfigurationError, "at least 1"):
            OllamaEmbeddingService(batch_size=0)
        with self.assertRaisesRegex(EmbeddingConfigurationError, "must not be empty"):
            OllamaEmbeddingService(model_name=" ")


if __name__ == "__main__":
    unittest.main()
