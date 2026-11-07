"""Initial LangGraph workflow foundation for AgentX."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from langgraph.graph import END, START, StateGraph

from agentx.agents.rag_agent import RAGAgent
from agentx.agents.report_generator import ReportGenerator
from agentx.agents.research_agent import ResearchAgent
from agentx.agents.verification_agent import VerificationAgent
from agentx.core.router import route_task
from agentx.core.state import AgentState, RouteName, SourceEvidence
from agentx.core.supervisor import classify_task


def _supervisor_node(_state: AgentState) -> dict[str, Any]:
    """Classify the request and record a clear workflow status."""
    query = _state.get("user_query", "")
    errors = list(_state.get("errors", []))

    if not isinstance(query, str) or not query.strip():
        return {
            "task_type": "unknown",
            "status": "invalid_query",
            "errors": errors + ["User query must not be empty."],
        }

    try:
        document_id = _state.get("document_id")
        task_type = (
            "document_qa"
            if (
                _state.get("force_knowledge") is True
                and isinstance(document_id, str)
                and document_id.strip()
            )
            else "combined_research"
            if (
                _state.get("force_combined") is True
                and isinstance(document_id, str)
                and document_id.strip()
            )
            else classify_task(query)
        )
    except Exception as exc:
        return {
            "task_type": "unknown",
            "status": "classification_error",
            "errors": errors + [f"Task classification failed ({type(exc).__name__})."],
        }

    status = "classified" if task_type != "unknown" else "classified_unknown"
    return {"task_type": task_type, "status": status, "errors": errors}


def _select_route(state: AgentState) -> RouteName:
    """Choose a route from shared state, defaulting safely to clarification."""
    return route_task(state.get("task_type"))


def _research_state_update(answer: Any) -> dict[str, Any]:
    """Convert a ResearchAnswer into the corresponding shared-state fields."""
    return {
        "research_summary": answer.summary,
        "research_results": [asdict(finding) for finding in answer.findings],
        "research_citations": [asdict(citation) for citation in answer.citations],
    }


def _web_source_evidence(answer: Any) -> list[SourceEvidence]:
    """Map real search citations and snippets into shared source records."""
    records: list[SourceEvidence] = []
    for index, citation in enumerate(answer.citations, 1):
        number = getattr(citation, "number", None)
        source_id = f"web:{number if number is not None else index}"
        records.append(
            {
                "source_id": source_id,
                "source_type": "web",
                "source_reference": getattr(citation, "url", None),
                "source_title": getattr(citation, "title", None),
                "text": getattr(citation, "snippet", None),
                "citation_id": number,
            }
        )
    return records


def _rag_state_update(answer: Any) -> dict[str, Any]:
    """Convert a RAGAnswer into answer, source, and citation state fields."""
    sources = [asdict(source) for source in answer.sources]
    citations = [
        {
            "number": source.number,
            "text": f"[{source.number}] {source.filename}, page {source.page_number}",
        }
        for source in answer.sources
    ]
    return {
        "rag_answer": answer.answer,
        "rag_status": answer.status,
        "retrieved_context": sources,
        "rag_citations": citations,
    }


def _document_source_evidence(answer: Any) -> list[SourceEvidence]:
    """Map retrieved PDF passage text and metadata into shared source records."""
    records: list[SourceEvidence] = []
    for index, source in enumerate(answer.sources, 1):
        document_id = getattr(source, "document_id", None)
        chunk_id = getattr(source, "chunk_id", None)
        number = getattr(source, "number", None)
        source_reference = chunk_id or document_id
        source_id = (
            f"document:{document_id or 'unknown'}:{chunk_id or number or index}"
        )
        records.append(
            {
                "source_id": source_id,
                "source_type": "document",
                "source_reference": source_reference,
                "source_title": getattr(source, "filename", None),
                "filename": getattr(source, "filename", None),
                "page_number": getattr(source, "page_number", None),
                "document_id": document_id,
                "chunk_id": chunk_id,
                "text": getattr(source, "text", None),
                "citation_id": number,
                "similarity": getattr(source, "similarity", None),
            }
        )
    return records


def _merge_source_evidence(
    existing: object, additions: list[SourceEvidence]
) -> list[SourceEvidence]:
    """Preserve existing evidence and make every source ID unique."""
    records: list[SourceEvidence] = []
    if isinstance(existing, (list, tuple)):
        records.extend(
            dict(item) for item in existing if isinstance(item, dict)
        )
    records.extend(additions)

    seen: set[str] = set()
    unique_records: list[SourceEvidence] = []
    for index, item in enumerate(records, 1):
        base_id = str(item.get("source_id") or f"source:{index}")
        source_id = base_id
        suffix = 2
        while source_id in seen:
            source_id = f"{base_id}:{suffix}"
            suffix += 1
        seen.add(source_id)
        item["source_id"] = source_id
        unique_records.append(item)  # type: ignore[arg-type]
    return unique_records


def _research_node_factory(research_agent: ResearchAgent):
    """Create a graph node that adapts ResearchAnswer into shared state."""

    def research_node(state: AgentState) -> dict[str, Any]:
        query = state.get("user_query", "")
        try:
            answer = research_agent.research(query)
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Research failed ({type(exc).__name__}): {exc}")
            return {
                "route": "research",
                "status": "research_error",
                "errors": errors,
            }

        update = {
            "route": "research",
            "status": (
                "research_no_sources"
                if answer.status == "no_sources"
                else "research_complete"
            ),
        }
        update.update(_research_state_update(answer))
        update["source_evidence"] = _merge_source_evidence(
            state.get("source_evidence"), _web_source_evidence(answer)
        )
        return update

    return research_node


def _knowledge_node_factory(rag_agent: RAGAgent | None):
    """Create a Knowledge node, initializing its RAG Agent only when used."""
    active_rag_agent = rag_agent

    def knowledge_node(state: AgentState) -> dict[str, Any]:
        nonlocal active_rag_agent
        if active_rag_agent is None:
            active_rag_agent = RAGAgent()

        query = state.get("user_query", "")
        document_id = state.get("document_id")
        try:
            answer = active_rag_agent.answer(query, document_id=document_id)
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Knowledge answering failed ({type(exc).__name__}): {exc}")
            return {
                "route": "knowledge",
                "status": "knowledge_error",
                "errors": errors,
            }

        update: dict[str, Any] = {
            "route": "knowledge",
            "status": {
                "success": "knowledge_complete",
                "insufficient_evidence": "knowledge_insufficient_evidence",
                "error": "knowledge_error",
            }.get(answer.status, "knowledge_error"),
        }
        update.update(_rag_state_update(answer))
        update["source_evidence"] = _merge_source_evidence(
            state.get("source_evidence"), _document_source_evidence(answer)
        )
        if answer.status == "error":
            errors = list(state.get("errors", []))
            errors.append(answer.error_message or "RAG Agent reported an unspecified error.")
            update["errors"] = errors
        return update

    return knowledge_node


def _combined_node_factory(
    research_agent: ResearchAgent,
    rag_agent: RAGAgent | None,
):
    """Create a node that runs web research and document QA independently."""
    active_rag_agent = rag_agent

    def combined_node(state: AgentState) -> dict[str, Any]:
        nonlocal active_rag_agent
        query = state.get("user_query", "")
        document_id = state.get("document_id")
        errors = list(state.get("errors", []))
        evidence_additions: list[SourceEvidence] = []
        research_failed = False
        rag_failed = False
        research_has_evidence = False
        rag_has_evidence = False
        update: dict[str, Any] = {"route": "combined"}

        # Run each agent separately so one failure never skips the other.
        try:
            research_answer = research_agent.research(query)
            update.update(_research_state_update(research_answer))
            evidence_additions.extend(_web_source_evidence(research_answer))
            research_has_evidence = research_answer.status != "no_sources"
        except Exception as exc:
            research_failed = True
            errors.append(f"Research failed ({type(exc).__name__}): {exc}")

        try:
            if active_rag_agent is None:
                active_rag_agent = RAGAgent()
            rag_answer = active_rag_agent.answer(query, document_id=document_id)
            update.update(_rag_state_update(rag_answer))
            evidence_additions.extend(_document_source_evidence(rag_answer))
            rag_has_evidence = rag_answer.status == "success"
            if rag_answer.status == "error":
                rag_failed = True
                errors.append(
                    rag_answer.error_message
                    or "RAG Agent reported an unspecified error."
                )
        except Exception as exc:
            rag_failed = True
            errors.append(f"Knowledge answering failed ({type(exc).__name__}): {exc}")

        if research_failed and rag_failed:
            status = "combined_failed"
        elif research_has_evidence and rag_has_evidence:
            status = "combined_complete"
        elif research_has_evidence or rag_has_evidence:
            status = "combined_partial_success"
        else:
            status = "combined_insufficient_evidence"

        update["status"] = status
        update["source_evidence"] = _merge_source_evidence(
            state.get("source_evidence"), evidence_additions
        )
        if errors:
            update["errors"] = errors
        return update

    return combined_node


def _clarification_node(_state: AgentState) -> dict[str, str]:
    """Placeholder for handling unclear or invalid requests."""
    return {"route": "clarification", "status": "clarification_required"}


def _verification_node_factory(verification_agent: VerificationAgent):
    """Create a final evidence-check node that preserves all original outputs."""

    def verification_node(state: AgentState) -> dict[str, Any]:
        try:
            report = verification_agent.verify(state)
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Verification failed ({type(exc).__name__}): {exc}")
            return {"verification_status": "error", "errors": errors}

        return {
            "verification_status": report.status,
            "verification_results": [asdict(item) for item in report.findings],
        }

    return verification_node


def _report_node_factory(report_generator: ReportGenerator):
    """Create a node that assembles a Markdown report from verified state."""

    def report_node(state: AgentState) -> dict[str, Any]:
        try:
            report = report_generator.generate(state)
        except Exception as exc:
            errors = list(state.get("errors", []))
            errors.append(f"Report generation failed ({type(exc).__name__}): {exc}")
            return {"report_status": "error", "errors": errors}
        return {"final_report": report.markdown, "report_status": report.status}

    return report_node


def build_workflow(
    *,
    research_agent: ResearchAgent | None = None,
    rag_agent: RAGAgent | None = None,
    verification_agent: VerificationAgent | None = None,
    report_generator: ReportGenerator | None = None,
) -> Any:
    """Build routes with optional research, knowledge, verification, and report agents."""
    active_research_agent = research_agent or ResearchAgent()
    active_verification_agent = verification_agent or VerificationAgent()
    active_report_generator = report_generator or ReportGenerator()
    graph = StateGraph(AgentState)
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("research", _research_node_factory(active_research_agent))
    graph.add_node("knowledge", _knowledge_node_factory(rag_agent))
    graph.add_node(
        "combined", _combined_node_factory(active_research_agent, rag_agent)
    )
    graph.add_node("clarification", _clarification_node)
    graph.add_node("verification", _verification_node_factory(active_verification_agent))
    graph.add_node("report", _report_node_factory(active_report_generator))
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        _select_route,
        {
            "research": "research",
            "knowledge": "knowledge",
            "combined": "combined",
            "clarification": "clarification",
        },
    )
    for node_name in ("research", "knowledge", "combined"):
        graph.add_edge(node_name, "verification")
    graph.add_edge("verification", "report")
    graph.add_edge("report", END)
    graph.add_edge("clarification", END)
    return graph.compile()
