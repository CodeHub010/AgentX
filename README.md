# AgentX

AgentX is an autonomous AI research and intelligence platform designed to combine web research, PDF question answering, CSV analysis, evidence verification, and report generation in one traceable workflow.

## Project status

This repository contains the project foundation, hybrid LLM gateway, research workflow, PDF ingestion, local embeddings, persistent vector storage, semantic retrieval, PDF RAG answer generation, and an initial LangGraph orchestration foundation. Further integrations will be added milestone by milestone.

## Architecture

```text
Streamlit UI (app.py) -> existing AgentX runner
                      |
                      v
             LangGraph workflow (agentx/core)
             |-- Web research and source collection
             |-- PDF ingestion -> ChromaDB retrieval
             |-- CSV ingestion -> Pandas analysis
             |-- Evidence verification and citation checks
             `-- Report synthesis
                      |
                      v
             Provider-neutral LLM gateway (agentx/llm)
                  |-- Ollama for local inference
                  `-- Optional OpenAI for cloud reasoning

SQLite: application metadata; ChromaDB: document vectors
Shared settings and schemas: agentx/core
```

The gateway returns a common chat interface (`invoke` and `ainvoke`) so agents do not import provider-specific classes. It loads provider SDKs only when selected. Cloud access is optional; local Ollama mode works without an OpenAI API key.

### Initial LangGraph orchestration

`agentx/core/state.py` defines the shared, progressively populated `AgentState`. `build_workflow()` in `agentx/core/workflow.py` starts at the supervisor, then uses conditional edges to select a route. The supervisor uses lightweight deterministic phrase cues to classify requests as `web_research`, `document_qa`, `combined_research`, or `unknown`. `agentx/core/router.py` maps those types to the `research`, `knowledge`, `combined`, or `clarification` route. The research route invokes the existing `ResearchAgent.research(query)` interface and stores its findings, summary, citations, and actual web snippets in shared state. The Knowledge route invokes the existing `RAGAgent.answer(question, document_id=...)` interface and stores its answer, retrieved PDF passage text, source metadata, and numbered citations. The Combined route runs Research first and RAG second, passing the same query and optional document ID, and stores evidence from each successful agent independently. It reports `combined_complete`, `combined_partial_success`, `combined_insufficient_evidence`, or `combined_failed` according to the outcomes. After Research, Knowledge, or Combined processing, the Verification Agent checks citation references and compares claims with preserved source text. Clarification remains unchanged. Invalid or missing task types safely use clarification.

### Evidence verification

`source_evidence` is a list of structured web and document records. Each record includes a workflow-unique `source_id`, source type, original citation identifier, reference (URL or document chunk), available title/file and page metadata, actual snippet or passage text, and similarity for PDF retrievals. Web snippets come from the configured search provider's results; PDF text comes from the retrieved chunks. Missing text remains missing rather than being reconstructed from generated answers. The combined path retains evidence from whichever agent succeeds.

### Markdown report generation

After evidence verification, the Research, Knowledge, and Combined routes pass the shared state to `ReportGenerator` in `agentx/agents/report_generator.py`. It deterministically assembles a Markdown report with the question, available web and PDF findings, verification labels, unsupported or conflicting claims, limitations, conclusion, and references. It uses only the preserved `source_evidence` records for source excerpts and citations, retaining original URLs, filenames, pages, IDs, and similarity metadata. Missing source text is called out explicitly, and generated findings without a matching original citation are not given a new citation.

Report assembly does not call an LLM, so it works in local/offline mode and does not add dependencies. `final_report` contains the Markdown and `report_status` is `generated`, `insufficient_evidence`, or `error`. The Clarification route remains unchanged and does not generate a report. Verification is currently deterministic and lexical; a `supported` label is not a guarantee of semantic entailment or factual truth.

### PDF report export

`PDFReportExporter` in `agentx/reports/pdf_exporter.py` converts the Markdown in `final_report` to a styled PDF. It formats headings, paragraphs, lists, block quotes, and Markdown tables, and keeps report text, original URLs, citations, filenames, and page references. It does not rewrite findings or create citations. Empty reports raise a clear `ValueError`; filesystem or PDF generation failures raise `PDFExportError`.

Install the added ReportLab dependency with `python -m pip install -r requirements.txt`. By default, PDFs are saved under the Git-ignored `reports/generated/` directory. Pass an optional safe filename and output directory when needed:

```python
from agentx.agents.report_generator import ReportGenerator
from agentx.reports import PDFReportExporter

report = ReportGenerator().generate(agent_state)
pdf_path = PDFReportExporter().export(report.markdown, filename="agentx-research.pdf")
print(pdf_path)
```

The exporter is a Markdown-to-PDF renderer; this step does not add export to the workflow or provide a command-line interface.

### End-to-end workflow smoke test

Run the complete local flow with one PowerShell command:

```powershell
python scripts/test_end_to_end.py --question "Research recent CNN-LSTM methods and compare them with the uploaded report." --document-id YOUR_STORED_DOCUMENT_ID
```

The runner validates a supplied document ID against the configured ChromaDB before using it. An explicit, available document ID routes the question through both web research and PDF RAG; without one, the existing Supervisor classification selects the appropriate route. It uses the local Ollama chat and embedding models and exports both Markdown and PDF files under `reports/generated/` unless `--report-dir` is set. `--persist-directory` can override the ChromaDB location and `--top-k` controls retrieved PDF chunks.

Search follows `WEB_SEARCH_PROVIDER`. Tavily is used only when selected and `TAVILY_API_KEY` is configured. If the key is missing, or a live Tavily request fails, the runner uses the repository's demo fixtures and explicitly labels the result as demo data, never live research. Keys are not printed. A RAG or Ollama failure is reported safely; the runner still saves a report when the workflow produced one, and marks the smoke command unsuccessful when an agent step recorded an error. Unit tests mock agent and storage boundaries and make no external calls.

`agentx/agents/verification_agent.py` checks citation IDs against the correct web or document source records, then compares claims with the actual cited snippets or passages. It uses sentence-level alignment, regular inflection normalization, and a small documented alias map for common equivalents (for example, “CNN” and “convolutional neural network,” or “capture” and “extract”). Missing citations and missing source text produce `insufficient_evidence`; an invalid citation ID or unrelated available cited text is `unsupported`. `conflicting` requires a sentence aligned to the same proposition with an explicit opposite polarity or incompatible numeric value. Source references retain URLs, filenames, pages, document IDs, and chunk IDs.

This is deterministic lexical matching, not semantic verification. It does not call an LLM, infer arbitrary paraphrases, judge source reliability, or establish factual truth. The alias list and sentence matching can miss valid paraphrases and can still misclassify nuanced language; ambiguous cases should remain insufficient rather than be treated as confirmed. Web snippets may be short excerpts rather than full pages.

### Streamlit research application

Launch the AgentX dashboard from the repository root:

```powershell
python -m streamlit run app.py
```

Enter a question and choose **Web research**, **Document only**, or **Combined research**. Document modes accept a PDF upload or a document previously indexed during the current browser session. Uploads are saved under `data/uploads/`, processed through the existing `PDFIngestor`, embedded in bounded batches with local Ollama, and upserted into the configured persistent ChromaDB. A content hash gives each PDF a stable document ID, so uploading it again updates the same chunk IDs. Document-only mode selects the Knowledge route; combined mode runs Research and RAG through the existing end-to-end runner.

The dashboard displays workflow and verification status, findings, preserved source snippets and references, the Markdown report, and a PDF download from the existing `PDFReportExporter`. Unsupported, conflicting, and insufficient evidence are highlighted. If Tavily is not configured, web search uses the repository's demo fixtures and the interface labels them as demo data. Ollama must be running with the configured chat and embedding models already installed; AgentX does not download models. Uploads are limited to 50 MB to keep memory use reasonable. The UI calls existing ingestion, storage, and workflow components and contains no duplicate agent logic.

## Resource strategy

- Target Python 3.11 or 3.12.
- On an 8 GB Windows machine, use a small quantized Ollama model and keep only one large model loaded at a time.
- Route only tasks that need complex reasoning to the optional cloud provider.
- Process PDFs and CSVs in bounded batches; avoid loading entire large corpora into memory.
- SQLite uses Python's standard library.

## Repository layout

```text
agentx/
  agents/       LangGraph workflows and agent nodes
  analysis/     Pandas-based dataset analysis
  api/          FastAPI routes and app assembly
  core/         Environment settings, schemas, domain utilities
  ingestion/    PDF and CSV loading/chunking
  llm/          Provider-neutral local/cloud model gateway
  reports/      Report assembly and rendering
  retrieval/    ChromaDB indexing and retrieval
  storage/      SQLite and persistence adapters
tests/          Standard-library unit tests
```

## Setup

1. Create and activate a Python 3.11 or 3.12 virtual environment.
2. Install dependencies when implementation work begins:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and configure only the providers you plan to use. Keep `.env` private; it is ignored by Git.
4. Install and start Ollama separately if using a local model. Ollama and its model weights are not Python dependencies. The gateway does not download models.

## PDF ingestion and chunking

`PDFIngestor` in `agentx/ingestion/pdf_ingestion.py` uses PyMuPDF to extract text page by page and produce overlapping character chunks. The defaults are 800 characters per chunk and 150 characters of overlap; configure them with `PDFIngestor(chunk_size=..., chunk_overlap=...)`. Chunks retain a generated or caller-supplied document ID, original filename, 1-based page number, and unique chunk ID. Empty pages are skipped; invalid, unsupported, encrypted, zero-byte, or all-text-empty documents produce clear ingestion errors. Scanned-image OCR is not included.

Generate a temporary sample PDF and run the ingestion demo, or pass a PDF path:

```powershell
python scripts/test_pdf_ingestion.py
python scripts/test_pdf_ingestion.py .\path\to\document.pdf
```

Install project dependencies with `python -m pip install -r requirements.txt`, or install only the PDF dependency with `python -m pip install "pymupdf>=1.24,<2"`. Run PDF-specific tests with `python -m unittest discover -s tests -p "test_pdf_ingestion.py" -v`.
## Hybrid LLM gateway configuration

`Settings.from_env()` reads environment variables and loads a local `.env` through `python-dotenv` when available. `create_chat_model()` uses `LLM_PROVIDER` unless an explicit provider is passed.

| Variable | Purpose | Default |
|---|---|---|
| `LLM_PROVIDER` | Default provider: `ollama` or `openai` | `ollama` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Local model tag | `llama3.2:1b` |
| `OPENAI_API_KEY` | Cloud API credential (required only for OpenAI) | unset |
| `OPENAI_MODEL` | Cloud model name | `gpt-4o-mini` |
| `LLM_CONNECT_TIMEOUT_SECONDS` | Connection timeout | `10` |
| `LLM_REQUEST_TIMEOUT_SECONDS` | Total request timeout | `120` |

Configuration errors identify invalid providers, timeouts, model settings, or a missing cloud key. OpenAI retries are capped at one retry.

## Local embedding generation

`OllamaEmbeddingService` in `agentx/embeddings/` accepts the existing `PDFChunk` records and calls the local Ollama `/api/embed` endpoint in bounded batches. It returns one `EmbeddedChunk` per input with the vector plus document ID, filename, page number, and chunk ID metadata. Defaults are `nomic-embed-text` and a batch size of 8. Use `iter_embeddings()` to stream output batch by batch, or `embed_chunks()` when collecting all results in memory is appropriate.

Set `OLLAMA_BASE_URL`, `OLLAMA_EMBEDDING_MODEL`, and optionally `OLLAMA_EMBEDDING_BATCH_SIZE` in `.env`. The service never downloads a model; if it is missing, the error shows the explicit `ollama pull` command. Timeout settings reuse `LLM_CONNECT_TIMEOUT_SECONDS` and `LLM_REQUEST_TIMEOUT_SECONDS`.

Run real local embedding generation with sample chunks, or provide a PDF path to use the PDF ingestion pipeline:

```powershell
python scripts/test_embeddings.py
python scripts/test_embeddings.py --pdf .\path\to\document.pdf
```

Run mocked HTTP unit tests (no Ollama connection):

```powershell
python -m unittest discover -s tests -p "test_embeddings.py" -v
```

## Persistent vector storage

`ChromaVectorStore` in `agentx/storage/` writes existing PDF chunks and their existing `EmbeddedChunk` vectors into a persistent local ChromaDB collection. It does not call Ollama or generate embeddings. Chunk IDs are Chroma record IDs; original text and document/file/page/chunk metadata are stored alongside each vector. `upsert` makes re-ingesting the same stable chunk IDs idempotent. The default database directory is `./data/chroma` (ignored by Git); override it with `CHROMA_PERSIST_DIRECTORY` or the constructor's `persist_directory` argument. Use `add_chunks(chunks, embeddings)` with matching ordered inputs. It writes bounded batches and returns the number of records written. `get_by_document_id(document_id)` returns stored text, metadata, and vectors for that document.

Install ChromaDB with `python -m pip install -r requirements.txt`. Test persistence in a temporary database and run the end-to-end local demo (which uses Ollama for embeddings):

```powershell
python -m unittest discover -s tests -p "test_vector_storage.py" -v
python scripts/test_vector_storage.py
python scripts/test_vector_storage.py --pdf .\path\to\document.pdf
```

## Semantic retrieval

`SemanticRetriever` in `agentx/retrieval/` finds stored PDF chunks that are semantically similar to a natural-language question. It embeds the question once with the existing `OllamaEmbeddingService` using the configured `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`), then passes that vector to the existing persistent ChromaDB collection. ChromaDB compares the query vector with stored chunk vectors using the collection's cosine distance. Results include the original text, document ID, filename, page number, chunk ID, distance, and cosine similarity. The default top-k is 5; pass `top_k` to the retriever or `--top-k` to the demo to change it, and optionally filter by `document_id`. Retrieval returns source chunks only; answer generation is not part of this step.

Run retrieval against the configured local database and Ollama service:

```powershell
python -m unittest discover -s tests -p "test_semantic_retrieval.py" -v
python scripts/test_semantic_retrieval.py
python scripts/test_semantic_retrieval.py --question "What are the main findings?" --top-k 3
python scripts/test_semantic_retrieval.py --question "Summarize the methods" --document-id YOUR_DOCUMENT_ID
```

The demo uses only records already stored in `CHROMA_PERSIST_DIRECTORY` (default `./data/chroma`). If the collection does not exist or has no matching chunks, it prints an ingestion hint; it does not create fixtures or insert sample results.

## PDF RAG answer generation

`RAGAgent` in `agentx/agents/rag_agent.py` combines the existing `SemanticRetriever` with the provider-neutral LLM Gateway. It embeds and retrieves the question, then gives the model only the retrieved passages with their real filename, page, document ID, chunk ID, and similarity score. The prompt instructs the model to use only this evidence, cite passages as `[1]`, `[2]`, and return `INSUFFICIENT_EVIDENCE:` when the passages do not support an answer. Unknown citation numbers are removed from the returned answer. The structured `RAGAnswer` includes the question, answer, status (`success`, `insufficient_evidence`, or `error`), and source metadata.

The default is 5 retrieved chunks. Configure the local chat and embedding models through `OLLAMA_MODEL` (default `llama3.2:1b`), `OLLAMA_EMBEDDING_MODEL` (default `nomic-embed-text`), and `OLLAMA_BASE_URL`; all existing timeout settings are reused. No additional RAG-specific dependency or API key is required in local Ollama mode. If retrieval returns no passages, the LLM is not called. This step returns answers and citations but does not independently verify each claim or guarantee model correctness; citations identify retrieved passages and should be checked against the source document for high-stakes use.

Run the local RAG demo against chunks already stored in ChromaDB:

```powershell
python -m unittest discover -s tests -p "test_rag_agent.py" -v
python scripts/test_rag_agent.py
python scripts/test_rag_agent.py --question "How does the CNN-LSTM architecture work?" --top-k 5 --document-id YOUR_DOCUMENT_ID
```

The demo accepts `--question`, `--top-k`, `--document-id`, and `--persist-directory`. It does not insert example documents or generate placeholder citations. Ingest a PDF and store its chunks before querying an empty database.
## Research agent and web search

`ResearchAgent` in `agentx/agents/` retrieves normalized sources, asks the existing LLM Gateway to synthesize only the supplied source snippets, and returns a structured answer with findings and numbered citations. Source metadata includes title, URL, snippet, and publication date when provided by the search backend. Snippets are bounded before prompt assembly to keep local model context use modest.

`WEB_SEARCH_PROVIDER=demo` is the default and uses clearly labeled example.com fixtures; it does not perform live web research. Configure Tavily for live search by setting `WEB_SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY` in `.env`. The Tavily adapter uses the Search API endpoint and bearer-token authentication. Live search has not been tested unless a real Tavily key is configured and a real search request is run.

Run the demonstration workflow (demo sources with the configured LLM, normally local Ollama):

```powershell
python scripts/test_research_agent.py
```

For real web results, first configure Tavily as above, then run the same command. A custom question can be passed as a quoted positional argument:

```powershell
python scripts/test_research_agent.py "What are recent advances in small language models?"
```
## Tests and verification

The tests use mocked provider constructors and the Python standard library. They do not make network calls, invoke paid APIs, or download models.

```powershell
python -m unittest discover -s tests -v
python -c "from agentx.core.settings import Settings; from agentx.llm import create_chat_model; print(Settings.from_env().llm_provider)"
```
