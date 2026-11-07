"""AgentX research dashboard. Launch with ``python -m streamlit run app.py``."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from agentx.core.end_to_end import (
    DocumentUnavailableError,
    EndToEndError,
    run_end_to_end,
)
from agentx.embeddings import EmbeddingError
from agentx.ingestion import PDFIngestionError
from agentx.storage import VectorStoreError
from agentx.ui.service import IngestedDocument, ingest_uploaded_pdf


st.set_page_config(page_title="AgentX Research", page_icon="🧭", layout="wide")


def _show_verification(state: dict) -> None:
    status = state.get("verification_status", "unavailable")
    label = {
        "supported": "Supported by available source text",
        "unsupported": "Unsupported claims detected",
        "conflicting": "Conflicting evidence detected",
        "insufficient_evidence": "Evidence is insufficient",
    }.get(status, str(status).replace("_", " ").title())
    if status in {"conflicting", "unsupported", "insufficient_evidence"}:
        st.warning(f"Verification: {label}")
    else:
        st.info(f"Verification: {label}")
    checks = state.get("verification_results") or []
    if isinstance(checks, list):
        for item in checks:
            if isinstance(item, dict):
                with st.expander(f"{item.get('status', 'unverified')}: {item.get('claim', 'Finding')}"):
                    st.write(item.get("explanation", "No explanation available."))
                    if item.get("supporting_sources"):
                        st.write("Sources:", item["supporting_sources"])


def _show_sources(state: dict) -> None:
    evidence = state.get("source_evidence") or []
    if not evidence:
        st.caption("No source evidence was returned.")
        return
    for item in evidence:
        if not isinstance(item, dict):
            continue
        title = item.get("source_title") or item.get("filename") or item.get("source_id") or "Source"
        with st.expander(str(title)):
            reference = item.get("source_reference")
            if reference:
                st.write(f"Reference: {reference}")
            if item.get("page_number") is not None:
                st.write(f"Page: {item['page_number']}")
            if item.get("document_id"):
                st.write(f"Document ID: {item['document_id']}")
            if item.get("chunk_id"):
                st.write(f"Chunk ID: {item['chunk_id']}")
            if item.get("citation_id") is not None:
                st.write(f"Citation: [{item['citation_id']}]")
            text = item.get("text")
            if text:
                st.write(text)
            else:
                st.caption("The source record contains no preserved snippet text.")


def _render_result(result) -> None:
    state = result.state
    st.subheader("Workflow results")
    cols = st.columns(3)
    cols[0].metric("Route", str(state.get("route", "unknown")).replace("_", " ").title())
    cols[1].metric("Workflow status", state.get("status", "unknown"))
    cols[2].metric("Report status", state.get("report_status", "unknown"))

    if result.search_mode in {"demo", "demo_fallback"} and state.get("route") in {"research", "combined"}:
        st.warning("Web-search sources below are DEMO fixture data, not live web research.")
    if result.notices:
        for notice in result.notices:
            st.info(notice)

    _show_verification(state)
    tabs = st.tabs(["Report", "Agent findings", "Sources & citations"])
    with tabs[0]:
        report = state.get("final_report") or "No Markdown report was produced."
        st.markdown(report)
        pdf_path = Path(result.pdf_path)
        if pdf_path.is_file():
            st.download_button(
                "Download PDF report",
                data=pdf_path.read_bytes(),
                file_name=pdf_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
        st.caption(f"Markdown: {result.markdown_path} · PDF: {result.pdf_path}")
    with tabs[1]:
        if state.get("research_summary"):
            st.markdown("#### Web research summary")
            st.write(state["research_summary"])
            for finding in state.get("research_results") or []:
                if isinstance(finding, dict):
                    st.write("• " + str(finding.get("text", finding)))
        if state.get("rag_answer"):
            st.markdown("#### Document answer")
            st.write(state["rag_answer"])
            st.caption(f"Document QA status: {state.get('rag_status', 'unknown')}")
        if not state.get("research_summary") and not state.get("rag_answer"):
            st.info("No agent findings were returned for this request.")
    with tabs[2]:
        _show_sources(state)
        citations = state.get("research_citations") or state.get("rag_citations") or []
        if citations:
            st.markdown("#### Citations")
            for citation in citations:
                st.write(citation)
    errors = state.get("errors") or []
    if errors:
        with st.expander("Workflow notices"):
            for error in errors:
                text = str(error).casefold()
                if any(token in text for token in ("ollama", "connection", "timeout", "model")):
                    st.write("A local model step failed. Confirm Ollama is running and the configured chat and embedding models are installed.")
                elif any(token in text for token in ("tavily", "search")):
                    st.write("A web search step failed. Demo fixtures may be used when live Tavily search is unavailable.")
                elif any(token in text for token in ("vector", "retriev", "chroma", "document")):
                    st.write("A document retrieval step failed. Confirm the PDF is indexed in the configured ChromaDB.")
                else:
                    st.write("An agent step reported an error; inspect the workflow status and source availability.")


st.title("🧭 AgentX")
st.caption("Research intelligence grounded in web sources and your documents")
st.markdown("Ask a question, choose a research mode, and review the evidence and verification before using the report.")

if "agentx_documents" not in st.session_state:
    st.session_state.agentx_documents = {}

with st.form("research_request"):
    question = st.text_area(
        "Research question",
        placeholder="For example: Explain the key methods and findings in this report.",
        height=120,
    )
    mode_label = st.selectbox(
        "Research mode",
        ["Web research", "Document only", "Combined research"],
        help="Web search may use demo fixtures when Tavily is not configured. Document modes require an uploaded or previously indexed PDF.",
    )
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    previous_docs = st.session_state.agentx_documents
    selected_doc = None
    if mode_label != "Web research" and previous_docs:
        selected_doc = st.selectbox(
            "Previously indexed document",
            [None, *previous_docs.keys()],
            format_func=lambda value: "Upload a new PDF" if value is None else f"{previous_docs[value]} · {value}",
        )
    top_k = st.slider("Document passages to retrieve", min_value=1, max_value=10, value=5)
    submitted = st.form_submit_button("Run AgentX research", type="primary", use_container_width=True)

if submitted:
    if not question.strip():
        st.error("Enter a research question to continue.")
    else:
        mode = {"Web research": None, "Document only": "document_only", "Combined research": "combined"}[mode_label]
        document_id = selected_doc
        try:
            if mode is not None and uploaded_pdf is not None:
                with st.status("Preparing PDF for local search…", expanded=True) as status:
                    status.write("Extracting pages, creating chunks, and indexing with local Ollama embeddings…")
                    indexed: IngestedDocument = ingest_uploaded_pdf(uploaded_pdf.getvalue(), uploaded_pdf.name)
                    st.session_state.agentx_documents[indexed.document_id] = indexed.filename
                    document_id = indexed.document_id
                    status.update(label=f"Indexed {indexed.filename} ({indexed.chunk_count} chunks)", state="complete")
            if mode is not None and not document_id:
                st.error("Upload a PDF or choose a previously indexed document for this mode.")
            else:
                with st.status("Running AgentX workflow…", expanded=True) as status:
                    def progress(message: str) -> None:
                        status.write(message)

                    result = run_end_to_end(
                        question,
                        document_id=document_id,
                        workflow_mode=mode,
                        top_k=top_k,
                        progress=progress,
                    )
                    status.update(label="Research workflow finished", state="complete")
                st.session_state.agentx_last_result = result
        except (PDFIngestionError, EmbeddingError, VectorStoreError, DocumentUnavailableError, EndToEndError, ValueError) as exc:
            message = str(exc)
            if isinstance(exc, EmbeddingError):
                message = "Could not connect to the configured local Ollama embedding model. Check that Ollama is running and the embedding model is installed."
            elif isinstance(exc, VectorStoreError):
                message = "Could not read or update the local ChromaDB. Check its configured directory and available disk space."
            elif isinstance(exc, PDFIngestionError):
                message = f"The PDF could not be processed: {exc}"
            elif isinstance(exc, EndToEndError):
                message = f"AgentX could not complete the workflow: {exc}"
            st.error(message)
        except Exception as exc:
            # Provider exception text may contain request details; surface only the type.
            st.error(f"AgentX encountered an unexpected {type(exc).__name__}. Check Ollama, ChromaDB, and provider configuration, then try again.")

if st.session_state.get("agentx_last_result") is not None:
    _render_result(st.session_state.agentx_last_result)

with st.sidebar:
    st.markdown("### Local setup")
    st.write("AgentX uses your configured Ollama models and local ChromaDB. No model is downloaded automatically.")
    st.write("Configure live web search with `WEB_SEARCH_PROVIDER=tavily` and `TAVILY_API_KEY`; without it, web results are labeled demo data.")
