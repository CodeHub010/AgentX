"""Tests for UI upload preparation without Ollama or ChromaDB calls."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentx.ingestion import PDFChunk, UnsupportedPDFError
from agentx.ui.service import ingest_uploaded_pdf, safe_upload_filename


class UploadServiceTests(unittest.TestCase):
    def test_upload_is_chunked_embedded_and_stored_with_stable_document_id(self):
        chunk = PDFChunk("doc", "report.pdf", 1, "doc:p1:c0001", "evidence")
        store = Mock()
        store.add_chunks.return_value = 1
        with tempfile.TemporaryDirectory() as directory:
            with (
            patch("agentx.ui.service.PDFIngestor") as ingestor_factory,
            patch("agentx.ui.service.OllamaEmbeddingService") as embedding_factory,
            patch("agentx.ui.service.ChromaVectorStore", return_value=store),
            patch("agentx.ui.service.local_runtime_settings", return_value=object()),
            ):
                ingestor_factory.return_value.ingest.return_value = [chunk]
                embedding_factory.return_value.iter_embeddings.return_value = iter([object()])
                result = ingest_uploaded_pdf(
                    b"pdf bytes", "My Report.pdf", upload_directory=Path(directory) / "uploads"
                )
                self.assertEqual(result.filename, "My Report.pdf")
                self.assertEqual(result.chunk_count, 1)
                self.assertTrue(result.path.is_file())
                self.assertEqual(ingestor_factory.return_value.ingest.call_args.kwargs["document_id"], result.document_id)
                store.add_chunks.assert_called_once_with([chunk], embedding_factory.return_value.iter_embeddings.return_value)
                store.close.assert_called_once()

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
