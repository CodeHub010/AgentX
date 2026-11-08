"""Tests for UI upload preparation without Ollama or ChromaDB calls."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentx.core.settings import Settings
from agentx.ingestion import PDFChunk, UnsupportedPDFError
from agentx.ui.service import ingest_uploaded_pdf, safe_upload_filename


class UploadServiceTests(unittest.TestCase):
    def test_upload_is_chunked_embedded_and_stored_with_stable_document_id(self):
        chunk = PDFChunk("doc", "report.pdf", 1, "doc:p1:c0001", "evidence")
        store = Mock()
        store.replace_document_chunks.return_value = 1
        with tempfile.TemporaryDirectory() as directory:
            with (
            patch("agentx.ui.service.PDFIngestor") as ingestor_factory,
            patch("agentx.ui.service.create_embedding_service") as embedding_factory,
            patch("agentx.ui.service.ChromaVectorStore", return_value=store) as store_factory,
            patch("agentx.ui.service.local_runtime_settings", return_value=object()),
            ):
                ingestor_factory.return_value.ingest.return_value = [chunk]
                embedding_factory.return_value.provider = "ollama"
                embedding_factory.return_value.model_name = "nomic-embed-text"
                embedding_factory.return_value.iter_embeddings.return_value = iter([object()])
                result = ingest_uploaded_pdf(
                    b"pdf bytes", "My Report.pdf", upload_directory=Path(directory) / "uploads"
                )
                self.assertEqual(result.filename, "My Report.pdf")
                self.assertEqual(result.chunk_count, 1)
                self.assertTrue(result.path.is_file())
                self.assertEqual(ingestor_factory.return_value.ingest.call_args.kwargs["document_id"], result.document_id)
                embedding_factory.assert_called_once_with(unittest.mock.ANY)
                self.assertEqual(
                    store_factory.call_args.kwargs["embedding_provider"], "ollama"
                )
                self.assertEqual(
                    store_factory.call_args.kwargs["embedding_model"], "nomic-embed-text"
                )
                store.replace_document_chunks.assert_called_once_with(
                    result.document_id, [chunk],
                    embedding_factory.return_value.iter_embeddings.return_value,
                )
                store.close.assert_called_once()

    def test_upload_uses_configured_gemini_embedding_identity(self):
        chunk = PDFChunk("doc", "report.pdf", 1, "doc:p1:c0001", "evidence")
        settings = Settings(
            embedding_provider="gemini",
            gemini_api_key="unit-test-key",
            gemini_embedding_model="gemini-embedding-001",
        )
        store = Mock()
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("agentx.ui.service.PDFIngestor") as ingestor_factory,
                patch("agentx.ui.service.create_embedding_service") as embedding_factory,
                patch("agentx.ui.service.ChromaVectorStore", return_value=store) as store_factory,
                patch("agentx.ui.service.local_runtime_settings", return_value=settings),
            ):
                ingestor_factory.return_value.ingest.return_value = [chunk]
                embedding_factory.return_value.provider = "gemini"
                embedding_factory.return_value.model_name = "gemini-embedding-001"
                embedding_factory.return_value.iter_embeddings.return_value = iter([object()])
                result = ingest_uploaded_pdf(
                    b"pdf bytes", "Gemini report.pdf", upload_directory=directory
                )
        embedding_factory.assert_called_once_with(settings)
        self.assertEqual(store_factory.call_args.kwargs["embedding_provider"], "gemini")
        self.assertEqual(
            store_factory.call_args.kwargs["embedding_model"], "gemini-embedding-001"
        )
        store.replace_document_chunks.assert_called_once_with(
            result.document_id, [chunk],
            embedding_factory.return_value.iter_embeddings.return_value,
        )

    def test_empty_upload_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            ingest_uploaded_pdf(b"", "report.pdf")

    def test_non_pdf_and_unsafe_names_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only PDF"):
            safe_upload_filename("notes.txt")
        with self.assertRaisesRegex(ValueError, "must not contain a path"):
            safe_upload_filename("..\\outside.pdf")

    def test_invalid_pdf_from_existing_ingestor_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("agentx.ui.service.PDFIngestor") as ingestor_factory:
                ingestor_factory.return_value.ingest.side_effect = UnsupportedPDFError("invalid PDF")
                with self.assertRaisesRegex(UnsupportedPDFError, "invalid PDF"):
                    ingest_uploaded_pdf(b"not a pdf", "report.pdf", upload_directory=directory)


if __name__ == "__main__":
    unittest.main()
