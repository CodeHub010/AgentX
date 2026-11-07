"""Deterministic mapping from classified task types to workflow routes."""

from __future__ import annotations

from agentx.core.state import RouteName


_TASK_ROUTES: dict[str, RouteName] = {
    "web_research": "research",
    "document_qa": "knowledge",
    "combined_research": "combined",
    "unknown": "clarification",
}


def route_task(task_type: object) -> RouteName:
    """Return the route for a task type; malformed values need clarification."""
    if not isinstance(task_type, str):
        return "clarification"
    return _TASK_ROUTES.get(task_type, "clarification")
