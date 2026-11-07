"""Small, testable services used by the Streamlit interface."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from agentx.core.end_to_end import local_runtime_settings
from agentx.core.settings import Settings
from agentx.embeddings import OllamaEmbeddingService
from agentx.ingestion import PDFChunk, PDFIngestor
from agentx.storage import ChromaVectorStore


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    """A PDF indexed in the configured local ChromaDB."""

    document_id: str
    filename: str
    chunk_count: int
    path: Path


def safe_upload_filename(filename: str) -> str:
    """Validate a portable PDF basename while retaining the original name."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("Choose a PDF with a valid filename.")
    name = filename.strip()
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("The uploaded filename must not contain a path.")
    if re.search(r'[<>:"|?*\x00-\x1f]', name) or name.endswith((".", " ")):
        raise ValueError("The uploaded filename contains unsupported characters.")
    if Path(name).suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")
    if Path(name).stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("The uploaded filename is reserved by Windows.")
    return name


def ingest_uploaded_pdf(
    content: bytes,
    filename: str,
    *,
    upload_directory: str | Path = "data/uploads",
    persist_directory: str | Path | None = None,
    settings: Settings | None = None,
) -> IngestedDocument:
    """Save, extract, embed, and persist an uploaded PDF using AgentX modules.

    The content hash provides a stable document ID, so uploading the same PDF
    again upserts the same chunk IDs. Embeddings are streamed to storage.
    """
    safe_name = safe_upload_filename(filename)
    if not isinstance(content, bytes) or not content:
        raise ValueError("The uploaded PDF is empty.")
    if len(content) > 50 * 1024 * 1024:
        raise ValueError("PDF uploads are limited to 50 MB on this lightweight app.")

    document_id = hashlib.sha256(content).hexdigest()[:24]
    path = Path(upload_directory) / document_id / safe_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

    chunks: list[PDFChunk] = PDFIngestor().ingest(path, document_id=document_id)
    runtime = local_runtime_settings(settings)
    embedding_service = OllamaEmbeddingService(settings=runtime)
    store = ChromaVectorStore(persist_directory)
    try:
        stored_count = store.add_chunks(chunks, embedding_service.iter_embeddings(chunks))
    finally:
        store.close()
    return IngestedDocument(document_id, safe_name, stored_count, path)
