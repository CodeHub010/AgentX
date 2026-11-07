"""Shared state schema for AgentX LangGraph workflows."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


TaskType = Literal["web_research", "document_qa", "combined_research", "unknown"]
RouteName = Literal["research", "knowledge", "combined", "clarification"]
EvidenceSourceType = Literal["web", "document"]


class SourceEvidence(TypedDict):
    """Original source content and its reference metadata."""

    source_id: str
    source_type: EvidenceSourceType
    source_reference: NotRequired[str | None]
    source_title: NotRequired[str | None]
    filename: NotRequired[str | None]
    page_number: NotRequired[int | None]
    document_id: NotRequired[str | None]
    chunk_id: NotRequired[str | None]
    text: NotRequired[str | None]
    citation_id: NotRequired[int | str | None]
    similarity: NotRequired[float | None]


class AgentState(TypedDict):
    """State passed between AgentX workflow nodes.

    Workflow nodes populate fields as work progresses, so only ``user_query``
    needs to be present when a new workflow is invoked.
    """

    user_query: str
    document_id: NotRequired[str | None]
    force_combined: NotRequired[bool]
    force_knowledge: NotRequired[bool]
    task_type: NotRequired[TaskType | None]
    route: NotRequired[RouteName | None]
    research_results: NotRequired[list[dict[str, object]] | None]
    research_summary: NotRequired[str | None]
    research_citations: NotRequired[list[dict[str, object]] | None]
    source_evidence: NotRequired[list[SourceEvidence] | None]
    retrieved_context: NotRequired[list[dict[str, object]] | None]
    rag_answer: NotRequired[str | None]
    rag_status: NotRequired[Literal["success", "insufficient_evidence", "error"] | None]
    rag_citations: NotRequired[list[dict[str, object]] | None]
    analysis_results: NotRequired[dict[str, object] | str | None]
    verification_results: NotRequired[
        list[dict[str, object]] | dict[str, object] | str | None
    ]
    verification_status: NotRequired[str | None]
    final_report: NotRequired[str | None]
    report_status: NotRequired[str | None]
    errors: NotRequired[list[str]]
    status: NotRequired[str]
