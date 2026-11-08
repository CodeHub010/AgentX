"""Persistence tests against isolated temporary ChromaDB databases."""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentx.embeddings import EmbeddedChunk
from agentx.ingestion import PDFChunk
from agentx.storage import (
    ChromaVectorStore,
    DuplicateChunkIDError,
    VectorStoreConfigurationError,
    VectorStoreDatabaseError,
)

try:
    import chromadb  # noqa: F401
except ImportError:
    chromadb = None


class FakeCollection:
    """Small collection fake for replacement ordering and scope checks."""

    def __init__(self, records=None, *, fail_upsert=False):
        self.records = dict(records or {})
        self.fail_upsert = fail_upsert
        self.delete_calls = []

    def get(self, *, where=None, ids=None, include=None):
        selected = list(ids) if ids is not None else list(self.records)
        selected = [chunk_id for chunk_id in selected if chunk_id in self.records]
        if where:
            selected = [
                chunk_id for chunk_id in selected
                if all(self.records[chunk_id]["metadata"].get(key) == value
                       for key, value in where.items())
            ]
        return {
            "ids": selected,
            "metadatas": [self.records[chunk_id]["metadata"] for chunk_id in selected],
        }

    def upsert(self, *, ids, documents, embeddings, metadatas):
        if self.fail_upsert:
            raise RuntimeError("simulated upsert failure")
        for chunk_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            self.records[chunk_id] = {
                "document": document, "embedding": embedding, "metadata": metadata
            }

    def delete(self, *, ids, where=None):
        self.delete_calls.append((list(ids), where))
        for chunk_id in ids:
            record = self.records.get(chunk_id)
            if record is not None and (
                not where or all(record["metadata"].get(key) == value for key, value in where.items())
            ):
                del self.records[chunk_id]


class ChromaVectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.store = ChromaVectorStore(
            Path(self.temp_dir.name) / "chroma", batch_size=2
        )
        self.addCleanup(self.store.close)

    @staticmethod
    def pair(chunk_id="doc-1:p1:c0001", text="Example PDF text."):
        chunk = PDFChunk("doc-1", "example.pdf", 1, chunk_id, text)
        embedded = EmbeddedChunk(
            "doc-1", "example.pdf", 1, chunk_id, (0.1, 0.2, 0.3)
        )
        return chunk, embedded

    @unittest.skipIf(chromadb is None, "ChromaDB dependency is not installed")
    def test_persists_content_vector_and_metadata_by_document(self):
        self.run_isolated("persistence", self.store.persist_directory)
        self.run_isolated("readback", self.store.persist_directory)

    @unittest.skipIf(chromadb is None, "ChromaDB dependency is not installed")
    def test_reingesting_same_ids_upserts_without_duplicates(self):
        self.run_isolated("reingestion", self.store.persist_directory)

    @unittest.skipIf(chromadb is None, "ChromaDB dependency is not installed")
    def test_replacement_removes_obsolete_chunks_without_touching_other_documents(self):
        self.run_isolated("replacement", self.store.persist_directory)

    @staticmethod
    def run_isolated(scenario, persist_directory):
        """Run native Chroma operations in a child process for Windows cleanup.

        Process exit releases native handles even on Chroma versions predating
        the public Client.close() API. The parent only deletes the temp folder
        after that process has fully exited.
        """
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--run-scenario",
                scenario,
                str(persist_directory),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if completed.returncode:
            raise AssertionError(
                f"Isolated Chroma scenario failed:\n{completed.stdout}\n{completed.stderr}"
            )

    def test_empty_inputs_write_nothing(self):
        self.assertEqual(self.store.add_chunks([], []), 0)

    def test_mismatched_input_lengths_are_rejected(self):
        chunk, vector = self.pair()
        with self.assertRaisesRegex(VectorStoreConfigurationError, "same number"):
            self.store.add_chunks([chunk], [])

    def test_duplicate_ids_in_one_input_are_rejected(self):
        chunk, vector = self.pair()
        with self.assertRaises(DuplicateChunkIDError):
            self.store.add_chunks([chunk, chunk], [vector, vector])

    def test_mismatched_chunk_and_vector_metadata_are_rejected(self):
        chunk, vector = self.pair()
        other = EmbeddedChunk("other", "example.pdf", 1, chunk.chunk_id, (0.1,))
        with self.assertRaisesRegex(VectorStoreConfigurationError, "metadata"):
            self.store.add_chunks([chunk], [other])

    def test_provider_model_mismatch_is_rejected_before_database_write(self):
        chunk = PDFChunk("doc-1", "example.pdf", 1, "doc-1:p1:c0001", "passage")
        vector = EmbeddedChunk(
            "doc-1", "example.pdf", 1, "doc-1:p1:c0001", (0.1, 0.2),
            provider="openai", model="text-embedding-3-small",
        )
        with self.assertRaisesRegex(VectorStoreConfigurationError, "identity mismatch"):
            self.store.add_chunks([chunk], [vector])
        self.assertIsNone(self.store._client)

    def test_inconsistent_or_nonfinite_embedding_dimensions_are_rejected(self):
        first_chunk, first_vector = self.pair("chunk-1")
        second_chunk, second_vector = self.pair("chunk-2")
        second_vector = EmbeddedChunk(
            "doc-1", "example.pdf", 1, "chunk-2", (0.1, 0.2)
        )
        with self.assertRaisesRegex(VectorStoreConfigurationError, "consistent dimensions"):
            self.store.add_chunks(
                [first_chunk, second_chunk], [first_vector, second_vector]
            )
        invalid_vector = EmbeddedChunk(
            "doc-1", "example.pdf", 1, first_chunk.chunk_id, (float("nan"),)
        )
        with self.assertRaisesRegex(VectorStoreConfigurationError, "invalid embedding values"):
            self.store.add_chunks([first_chunk], [invalid_vector])

    def test_non_default_provider_model_gets_separate_collection(self):
        openai_store = ChromaVectorStore(
            Path(self.temp_dir.name) / "chroma",
            embedding_provider="openai",
            embedding_model="text-embedding-3-small",
        )
        gemini_store = ChromaVectorStore(
            Path(self.temp_dir.name) / "chroma",
            embedding_provider="gemini",
            embedding_model="gemini-embedding-001",
        )
        try:
            self.assertNotEqual(self.store.collection_name, openai_store.collection_name)
            self.assertNotEqual(self.store.collection_name, gemini_store.collection_name)
            self.assertNotEqual(openai_store.collection_name, gemini_store.collection_name)
            self.assertEqual(self.store.collection_name, "agentx_pdf_chunks")
        finally:
            openai_store.close()
            gemini_store.close()

    def test_document_replacement_removes_only_its_obsolete_chunks(self):
        old_chunk, _ = self.pair("doc-1:p1:c0001", "Old page one")
        stale_chunk, _ = self.pair("doc-1:p2:c0001", "Old page two")
        other = PDFChunk("doc-2", "other.pdf", 1, "doc-2:p1:c0001", "Other document")
        collection = FakeCollection({
            chunk.chunk_id: {"metadata": {
                **chunk.metadata,
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
            }}
            for chunk in (old_chunk, stale_chunk, other)
        })
        new_chunk = PDFChunk("doc-1", "example.pdf", 1, "doc-1:p1:c0001", "New page one")
        new_vector = EmbeddedChunk(
            "doc-1", "example.pdf", 1, new_chunk.chunk_id, (0.2, 0.3, 0.4)
        )
        with patch.object(self.store, "_get_collection", return_value=collection):
            written = self.store.replace_document_chunks("doc-1", [new_chunk], [new_vector])

        self.assertEqual(written, 1)
        self.assertEqual(set(collection.records), {new_chunk.chunk_id, other.chunk_id})
        self.assertEqual(
            collection.delete_calls,
            [([stale_chunk.chunk_id], {"document_id": "doc-1"})],
        )

    def test_failed_document_replacement_upsert_does_not_clean_up(self):
        old_chunk, _ = self.pair("doc-1:p1:c0001", "Old text")
        collection = FakeCollection({
            old_chunk.chunk_id: {"metadata": {
                **old_chunk.metadata,
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
            }}
        }, fail_upsert=True)
        new_chunk = PDFChunk("doc-1", "example.pdf", 1, "doc-1:p2:c0001", "New text")
        new_vector = EmbeddedChunk(
            "doc-1", "example.pdf", 1, new_chunk.chunk_id, (0.2, 0.3, 0.4)
        )
        with patch.object(self.store, "_get_collection", return_value=collection):
            with self.assertRaisesRegex(VectorStoreDatabaseError, "could not store a batch"):
                self.store.replace_document_chunks("doc-1", [new_chunk], [new_vector])

        self.assertEqual(collection.delete_calls, [])
        self.assertEqual(set(collection.records), {old_chunk.chunk_id})

    def test_failed_embedding_generation_does_not_clean_up(self):
        old_chunk, _ = self.pair("doc-1:p1:c0001", "Old text")
        collection = FakeCollection({
            old_chunk.chunk_id: {"metadata": {
                **old_chunk.metadata,
                "embedding_provider": "ollama",
                "embedding_model": "nomic-embed-text",
            }}
        })
        new_chunk = PDFChunk("doc-1", "example.pdf", 1, "doc-1:p2:c0001", "New text")

        def fail_embedding():
            raise RuntimeError("simulated embedding failure")
            yield  # Make this an iterator while preserving the failure.

        with patch.object(self.store, "_get_collection", return_value=collection):
            with self.assertRaisesRegex(RuntimeError, "embedding failure"):
                self.store.replace_document_chunks("doc-1", [new_chunk], fail_embedding())

        self.assertEqual(collection.delete_calls, [])
        self.assertEqual(set(collection.records), {old_chunk.chunk_id})

    def test_document_replacement_rejects_chunks_from_another_document(self):
        other_chunk = PDFChunk("doc-2", "other.pdf", 1, "doc-2:p1:c1", "Other")
        with self.assertRaisesRegex(VectorStoreConfigurationError, "must belong"):
            self.store.replace_document_chunks("doc-1", [other_chunk], [])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--run-scenario":
        scenario, directory = sys.argv[2], Path(sys.argv[3])
        store = ChromaVectorStore(directory, batch_size=2)
        first = ChromaVectorStoreTests.pair()
        if scenario == "persistence":
            second = ChromaVectorStoreTests.pair(
                "doc-1:p2:c0001", "Second page content."
            )
            assert store.add_chunks(
                [first[0], second[0]], [first[1], second[1]]
            ) == 2
            records = store.get_by_document_id("doc-1")
            assert len(records) == 2
            record = next(item for item in records if item.chunk_id == first[0].chunk_id)
            assert record.text == first[0].text
            assert record.filename == "example.pdf"
            assert record.page_number == 1
            assert record.embedding_provider == "ollama"
            assert record.embedding_model == "nomic-embed-text"
            assert len(record.embedding) == len(first[1].embedding)
            assert all(
                math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
                for actual, expected in zip(record.embedding, first[1].embedding)
            )
            store.close()
        elif scenario == "readback":
            records = store.get_by_document_id("doc-1")
            assert len(records) == 2
            record = next(item for item in records if item.chunk_id == first[0].chunk_id)
            assert record.text == first[0].text
            assert record.embedding_provider == "ollama"
            assert record.embedding_model == "nomic-embed-text"
            assert all(
                math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-6)
                for actual, expected in zip(record.embedding, first[1].embedding)
            )
            store.close()
        elif scenario == "reingestion":
            store.add_chunks([first[0]], [first[1]])
            store.add_chunks([first[0]], [first[1]])
            assert len(store.get_by_document_id("doc-1")) == 1
            store.close()
        elif scenario == "replacement":
            second = ChromaVectorStoreTests.pair(
                "doc-1:p2:c0001", "Second page content."
            )
            other_chunk = PDFChunk(
                "doc-2", "other.pdf", 1, "doc-2:p1:c0001", "Other document."
            )
            other_vector = EmbeddedChunk(
                "doc-2", "other.pdf", 1, other_chunk.chunk_id, (0.2, 0.3, 0.4)
            )
            store.add_chunks(
                [first[0], second[0], other_chunk],
                [first[1], second[1], other_vector],
            )
            store.replace_document_chunks("doc-1", [first[0]], [first[1]])
            assert len(store.get_by_document_id("doc-1")) == 1
            other_records = store.get_by_document_id("doc-2")
            assert len(other_records) == 1
            assert other_records[0].text == "Other document."
            store.close()
        else:
            raise ValueError(f"Unknown test scenario {scenario!r}")
    else:
        unittest.main()
