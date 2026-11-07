"""Lightweight deterministic task classification for the workflow supervisor."""

from __future__ import annotations

import re

from agentx.core.state import TaskType


_DOCUMENT_CUES = (
    r"\b(document|pdf|report|uploaded file|uploaded document|the file|this file)\b",
    r"\b(in|from|inside|according to|based on) (the|this|my) (document|pdf|report|file)\b",
)
_WEB_CUES = (
    r"\b(web|internet|online|website|websites|sources|citations|references)\b",
    r"\b(research|look up|search|find|latest|current|recent)\b",
)


def classify_task(user_query: str) -> TaskType:
    """Classify a query using explicit intent phrases, without external calls.

    A request that explicitly asks to combine a document with web research is
    classified as ``combined_research``. Clear document-only and web-only
    requests map to their corresponding types. Unclear or empty requests map
    to ``unknown`` so the workflow does not assume a source or invent a file.
    """
    if not isinstance(user_query, str) or not user_query.strip():
        return "unknown"

    query = re.sub(r"\s+", " ", user_query.casefold()).strip()
    has_document_intent = any(re.search(pattern, query) for pattern in _DOCUMENT_CUES)
    has_web_intent = any(re.search(pattern, query) for pattern in _WEB_CUES)

    if has_document_intent and has_web_intent:
        return "combined_research"
    if has_document_intent:
        return "document_qa"
    if has_web_intent:
        return "web_research"
    return "unknown"
