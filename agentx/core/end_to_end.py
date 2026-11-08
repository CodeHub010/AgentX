"""Run the existing AgentX agents, verification, report, and PDF export."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from agentx.agents import RAGAgent, ResearchAgent, ReportGenerator
from agentx.core.settings import ConfigurationError, Settings
from agentx.core.workflow import build_workflow
from agentx.embeddings import create_embedding_service
from agentx.retrieval import SemanticRetriever
from agentx.reports import PDFReportExporter
from agentx.storage import ChromaVectorStore, VectorStoreError
from agentx.tools.web_search import (
    DemoSearchProvider,
    SearchProviderError,
    create_web_search_provider,
)


class EndToEndError(RuntimeError):
    """Raised when the end-to-end run cannot continue safely."""


class DocumentUnavailableError(EndToEndError):
    """A requested document ID is absent from the configured Chroma database."""


@dataclass(frozen=True, slots=True)
class EndToEndResult:
    """Result of one workflow run and its Markdown/PDF outputs."""

    state: dict[str, Any]
    markdown_path: Path
    pdf_path: Path
    search_mode: str
    document_id: str | None
    notices: tuple[str, ...] = ()


class _FallbackResearchAgent:
    """Use demo fixtures only when configured live Tavily search fails."""

    def __init__(self, primary: ResearchAgent, settings: Settings) -> None:
        self._primary = primary
        self._fallback = ResearchAgent(
            settings=replace(settings, web_search_provider="demo", tavily_api_key=None),
            search_provider=DemoSearchProvider(),
        )
        self.search_mode = "tavily"
        self.fallback_error_type: str | None = None

    def research(self, query: str, **kwargs: Any):
        try:
            return self._primary.research(query, **kwargs)
        except SearchProviderError as exc:
            self.search_mode = "demo_fallback"
            self.fallback_error_type = type(exc).__name__
            return self._fallback.research(query, **kwargs)


def _runtime_settings(settings: Settings | None) -> tuple[Settings, list[str]]:
    """Load validated runtime settings, including the selected chat provider.

    Tavily without a key falls back to labeled demo fixtures. Credential values
    are never returned or logged.
    """
    selected, notices = _load_runtime_configuration(settings)
    if selected.llm_provider == "openai" and not selected.openai_api_key:
        raise EndToEndError(
            "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not configured. "
            "Set the key or choose LLM_PROVIDER=ollama."
        )
    if selected.llm_provider == "gemini" and not selected.gemini_api_key:
        raise EndToEndError(
            "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not configured. "
            "Set it as an environment variable or a top-level Streamlit secret."
        )
    return selected, notices


def _load_runtime_configuration(settings: Settings | None) -> tuple[Settings, list[str]]:
    """Read base settings and preserve the safe demo-search fallback."""
    notices: list[str] = []
    if settings is None:
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        environment = dict(os.environ)
        requested_search = environment.get("WEB_SEARCH_PROVIDER", "demo").strip().lower()
        has_tavily_key = bool(environment.get("TAVILY_API_KEY", "").strip())
        if requested_search == "tavily" and not has_tavily_key:
            environment["WEB_SEARCH_PROVIDER"] = "demo"
            notices.append("TAVILY_API_KEY is not configured; using labeled demo search data.")
        try:
            settings = Settings.from_env(environment)
        except ConfigurationError as exc:
            raise EndToEndError(f"AgentX configuration is invalid: {exc}") from exc

    selected = settings
    if selected.web_search_provider == "tavily" and not selected.tavily_api_key:
        selected = replace(selected, web_search_provider="demo")
        notices.append("TAVILY_API_KEY is not configured; using labeled demo search data.")
    try:
        selected.validate(provider=selected.llm_provider)
    except ConfigurationError as exc:
        raise EndToEndError(f"AgentX configuration is invalid: {exc}") from exc
    return selected, notices


def local_runtime_settings(settings: Settings | None = None) -> Settings:
    """Return local Ollama settings for the embedding-only upload pipeline."""
    configured, _notices = _load_runtime_configuration(settings)
    local_settings = replace(configured, llm_provider="ollama")
    try:
        local_settings.validate(provider="ollama")
    except ConfigurationError as exc:
        raise EndToEndError(f"Local Ollama configuration is invalid: {exc}") from exc
    return local_settings


def run_end_to_end(
    question: str,
    *,
    document_id: str | None = None,
    top_k: int = 5,
    persist_directory: str | Path | None = None,
    report_output_dir: str | Path | None = None,
    workflow_mode: str | None = None,
    settings: Settings | None = None,
    progress: Callable[[str], None] | None = print,
) -> EndToEndResult:
    """Run the graph with the configured chat provider and PDF retrieval/export.

    This function uses existing AgentX agent interfaces. Tests can inject
    ``settings`` and replace module constructors with mocks; no external calls
    are made until the actual Research/RAG agents execute.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Research question must not be empty.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    if document_id is not None and (not isinstance(document_id, str) or not document_id.strip()):
        raise ValueError("document_id must contain non-whitespace text when provided.")
    if workflow_mode not in {None, "combined", "document_only"}:
        raise ValueError("workflow_mode must be 'combined', 'document_only', or None.")
    if workflow_mode is not None and document_id is None:
        raise ValueError("workflow_mode requires a validated document_id.")

    notify = progress or (lambda _message: None)
    notify("[1/5] Loading model provider and search configuration...")
    configured, notices = _runtime_settings(settings)

    selected_document_id = document_id.strip() if document_id else None
    embedding_service = None
    store = None
    try:
        if selected_document_id:
            notify("[2/5] Opening the configured ChromaDB...")
            try:
                embedding_service = create_embedding_service(configured)
            except Exception as exc:
                raise EndToEndError(
                    f"Embedding provider configuration failed ({type(exc).__name__}). "
                    "Check embedding provider settings and credentials."
                ) from exc
            store = ChromaVectorStore(
                persist_directory,
                embedding_provider=embedding_service.provider,
                embedding_model=embedding_service.model_name,
            )
            try:
                chunks = store.get_by_document_id(selected_document_id)
            except VectorStoreError as exc:
                raise DocumentUnavailableError(
                    "The requested document could not be read from the configured ChromaDB. "
                    "Confirm the document ID and database directory."
                ) from exc
            if not chunks:
                raise DocumentUnavailableError(
                    f"Document ID {selected_document_id!r} has no stored chunks in this ChromaDB."
                )
            filename = chunks[0].filename
            notify(f"Validated document: {filename} ({len(chunks)} stored chunks).")

        try:
            search_provider = create_web_search_provider(configured)
        except ConfigurationError as exc:
            raise EndToEndError(f"Search provider configuration is invalid: {exc}") from exc
        if configured.web_search_provider == "tavily":
            notify("Search mode: live Tavily (API key configured; key value hidden).")
            primary_research = ResearchAgent(
                settings=configured, search_provider=search_provider
            )
            research_agent: Any = _FallbackResearchAgent(primary_research, configured)
        else:
            notify("Search mode: DEMO fixtures; these are not live web research.")
            research_agent = ResearchAgent(
                settings=configured, search_provider=search_provider
            )

        rag_agent = None
        if embedding_service is not None and store is not None:
            retriever = SemanticRetriever(
                embedding_service=embedding_service,
                vector_store=store,
                settings=configured,
                top_k=top_k,
            )
            rag_agent = RAGAgent(
                settings=configured,
                retriever=retriever,
                top_k=top_k,
            )

        notify("[3/5] Running Supervisor, route, Research/RAG, and Verification nodes...")
        workflow = build_workflow(
            research_agent=research_agent, rag_agent=rag_agent, settings=configured
        )
        state = workflow.invoke(
            {
                "user_query": question.strip(),
                "document_id": selected_document_id,
                "force_combined": selected_document_id is not None and workflow_mode != "document_only",
                "force_knowledge": selected_document_id is not None and workflow_mode == "document_only",
            }
        )
        route = state.get("route", "unknown")
        notify(
            f"Workflow finished: route={route}, status={state.get('status', 'unknown')}, "
            f"verification={state.get('verification_status', 'unavailable')}."
        )

        if isinstance(research_agent, _FallbackResearchAgent):
            if research_agent.search_mode == "demo_fallback":
                notices.append(
                    "Live Tavily search failed; the report uses labeled demo search fixtures instead."
                )
                notify("Tavily request failed; continuing with DEMO fixtures (not live research).")
            search_mode = research_agent.search_mode
        else:
            search_mode = configured.web_search_provider
        for notice in notices:
            notify(f"Notice: {notice}")

        markdown = state.get("final_report")
        if not isinstance(markdown, str) or not markdown.strip():
            raise EndToEndError(
                "The workflow did not produce a report (for example, the question may need clarification)."
            )

        if search_mode in {"demo", "demo_fallback"} and route in {"research", "combined"}:
            markdown = (
                "> **Search provenance: DEMO fixture results; these are not live web research.**\n\n"
                + markdown
            )
            state["final_report"] = markdown

        # Redact configured secret values from any provider exception text that
        # may have been carried into report limitations. Findings and citations
        # are otherwise preserved byte-for-byte.
        markdown = _redact_report_secrets(markdown, configured)
        notify("[4/5] Saving Markdown and exporting the report as PDF...")
        exporter = PDFReportExporter(report_output_dir)
        pdf_path = exporter.export(markdown)
        markdown_path = pdf_path.with_suffix(".md")
        markdown_path.write_text(markdown, encoding="utf-8")
        notify(f"Markdown report: {markdown_path.resolve()}")
        notify(f"PDF report: {pdf_path.resolve()}")
        notify(f"Report status: {state.get('report_status', 'unknown')}.")
        if state.get("report_status") == "insufficient_evidence":
            notify("Evidence warning: the report was generated, but its available source text was insufficient for verification.")
        if state.get("rag_status") == "insufficient_evidence":
            notify("Document QA warning: retrieved PDF evidence was insufficient to answer the question.")
        _notify_runtime_errors(state, configured, notify)
        notify("[5/5] End-to-end smoke test completed.")
        return EndToEndResult(
            state=dict(state),
            markdown_path=markdown_path,
            pdf_path=pdf_path,
            search_mode=search_mode,
            document_id=selected_document_id,
            notices=tuple(notices),
        )
    finally:
        if store is not None:
            store.close()


def _redact_report_secrets(markdown: str, settings: Settings) -> str:
    """Mask configured credential values if an exception echoed one."""
    safe_markdown = markdown
    for secret in (settings.tavily_api_key, settings.openai_api_key, settings.gemini_api_key):
        if secret:
            safe_markdown = safe_markdown.replace(secret, "[REDACTED]")
    return safe_markdown


def _notify_runtime_errors(
    state: dict[str, Any], settings: Settings, notify: Callable[[str], None]
) -> None:
    """Report safe error summaries without printing exception text or secrets."""
    errors = state.get("errors")
    if not isinstance(errors, list) or not errors:
        return
    for error in errors:
        text = str(error).casefold()
        if any(token in text for token in ("ollama", "connection", "timeout", "model", "gemini", "openai")):
            notify("Model provider warning: check the selected provider, credentials, quota, model name, and connectivity.")
        elif "tavily" in text or "search" in text:
            notify("Search warning: the configured search request did not complete successfully.")
        elif any(token in text for token in ("vector", "retriev", "pdf", "collection")):
            notify("PDF retrieval warning: stored document evidence may be unavailable or insufficient.")
        else:
            notify("Workflow warning: one or more agent steps reported an error; see workflow state for details.")

