"""Unit tests for semantic retrieval with mocked Ollama and Chroma adapters."""

import unittest
from unittest.mock import Mock

from agentx.embeddings import (
    EmbeddedChunk,
    EmbeddingConnectionError,
)
from agentx.storage import (
    StoredChunk,
    VectorStoreCollectionNotFoundError,
    VectorStoreDatabaseError,
)
from agentx.retrieval import (
    EmptyQuestionError,
    MissingVectorCollectionError,
    RetrievalConfigurationError,
    RetrievalEmbeddingError,
    RetrievalStorageError,
    SemanticRetriever,
)


class SemanticRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.embedding_service = Mock()
        self.vector_store = Mock()
        self.query_vector = (0.25, 0.5, 0.75)
        self.embedding_service.embed_chunks.return_value = [
            EmbeddedChunk(
                "__agentx_query__",
                "query.txt",
                0,
                "__agentx_query__",
                self.query_vector,
            )
        ]
        self.retriever = SemanticRetriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
        )

    @staticmethod
    def stored_chunk(
        *, document_id="doc-1", filename="paper.pdf", page_number=4,
        chunk_id="doc-1:p4:c0002", text="Relevant source passage.", distance=0.12,
    ):
        return StoredChunk(
            document_id=document_id,
            filename=filename,
            page_number=page_number,
            chunk_id=chunk_id,
            text=text,
            embedding=(),
            distance=distance,
        )

    def test_successful_retrieval_returns_source_and_similarity_data(self):
        self.vector_store.query_by_embedding.return_value = [self.stored_chunk()]
        results = self.retriever.retrieve("What does the paper find?")

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.text, "Relevant source passage.")
        self.assertEqual(result.document_id, "doc-1")
        self.assertEqual(result.filename, "paper.pdf")
        self.assertEqual(result.page_number, 4)
        self.assertEqual(result.chunk_id, "doc-1:p4:c0002")
        self.assertAlmostEqual(result.distance, 0.12)
        self.assertAlmostEqual(result.similarity, 0.88)

    def test_question_is_embedded_once_then_same_vector_is_queried(self):
        self.vector_store.query_by_embedding.return_value = []
        self.retriever.retrieve("  How does retrieval work?  ")

        query_chunks = self.embedding_service.embed_chunks.call_args.args[0]
        self.assertEqual(len(query_chunks), 1)
        self.assertEqual(query_chunks[0].text, "How does retrieval work?")
        self.vector_store.query_by_embedding.assert_called_once_with(
            self.query_vector, top_k=5, document_id=None
        )

    def test_top_k_defaults_and_can_be_configured_per_request(self):
        self.vector_store.query_by_embedding.return_value = []
        configured = SemanticRetriever(
            embedding_service=self.embedding_service,
            vector_store=self.vector_store,
            top_k=3,
        )
        configured.retrieve("question", top_k=7)
        self.vector_store.query_by_embedding.assert_called_once_with(
            self.query_vector, top_k=7, document_id=None
        )

        self.vector_store.query_by_embedding.reset_mock()
        self.retriever.retrieve("question")
        self.vector_store.query_by_embedding.assert_called_once_with(
            self.query_vector, top_k=5, document_id=None
        )

    def test_injected_store_must_match_explicit_embedding_identity(self):
        self.embedding_service.provider = "openai"
        self.embedding_service.model_name = "text-embedding-3-small"
        self.vector_store.embedding_provider = "ollama"
        self.vector_store.embedding_model = "nomic-embed-text"
        with self.assertRaisesRegex(RetrievalConfigurationError, "do not match"):
            SemanticRetriever(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store,
            )

    def test_document_id_filter_is_forwarded(self):
        self.vector_store.query_by_embedding.return_value = []
        self.retriever.retrieve("question", document_id="doc-42")
        self.vector_store.query_by_embedding.assert_called_once_with(
            self.query_vector, top_k=5, document_id="doc-42"
        )

    def test_empty_question_is_rejected_before_external_calls(self):
        with self.assertRaises(EmptyQuestionError):
            self.retriever.retrieve("  \n")
        self.embedding_service.embed_chunks.assert_not_called()
        self.vector_store.query_by_embedding.assert_not_called()

    def test_embedding_service_errors_are_reported(self):
        self.embedding_service.embed_chunks.side_effect = EmbeddingConnectionError(
            "Ollama unavailable"
        )
        with self.assertRaisesRegex(RetrievalEmbeddingError, "Ollama unavailable"):
            self.retriever.retrieve("question")
        self.vector_store.query_by_embedding.assert_not_called()

    def test_empty_retrieval_results_are_returned_as_empty_list(self):
        self.vector_store.query_by_embedding.return_value = []
        self.assertEqual(self.retriever.retrieve("question"), [])

    def test_missing_collection_has_actionable_error(self):
        self.vector_store.query_by_embedding.side_effect = (
            VectorStoreCollectionNotFoundError("missing")
        )
        with self.assertRaisesRegex(MissingVectorCollectionError, "store a PDF"):
            self.retriever.retrieve("question")

    def test_storage_failures_are_reported(self):
        self.vector_store.query_by_embedding.side_effect = VectorStoreDatabaseError(
            "database offline"
        )
        with self.assertRaisesRegex(RetrievalStorageError, "database offline"):
            self.retriever.retrieve("question")

    def test_invalid_top_k_configuration_is_rejected(self):
        for invalid in (0, -1, True, 1.5, "5"):
            with self.subTest(top_k=invalid):
                with self.assertRaises(RetrievalConfigurationError):
                    SemanticRetriever(
                        embedding_service=self.embedding_service,
                        vector_store=self.vector_store,
                        top_k=invalid,
                    )
        with self.assertRaises(RetrievalConfigurationError):
            self.retriever.retrieve("question", top_k=0)

    def test_blank_document_filter_is_rejected(self):
        with self.assertRaises(RetrievalConfigurationError):
            self.retriever.retrieve("question", document_id=" ")


if __name__ == "__main__":
    unittest.main()
