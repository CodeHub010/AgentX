"""Persistence tests against isolated temporary ChromaDB databases."""

import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentx.embeddings import EmbeddedChunk
from agentx.ingestion import PDFChunk
from agentx.storage import (
    ChromaVectorStore,
    DuplicateChunkIDError,
    VectorStoreConfigurationError,
)

try:
    import chromadb  # noqa: F401
except ImportError:
    chromadb = None


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
        else:
            raise ValueError(f"Unknown test scenario {scenario!r}")
    else:
        unittest.main()
