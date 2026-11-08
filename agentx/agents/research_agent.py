"""Research workflow: retrieve sources, synthesize with the configured LLM, cite."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agentx.core.settings import Settings
from agentx.llm import ChatModel, create_chat_model
from agentx.tools.web_search import (
    SearchProviderError,
    WebSearchProvider,
    WebSource,
    create_web_search_provider,
)


@dataclass(frozen=True, slots=True)
class ResearchCitation:
    """Citation metadata for one retrieved source."""

    number: int
    title: str
    url: str
    published_date: str | None = None
    snippet: str | None = None


@dataclass(frozen=True, slots=True)
class ResearchFinding:
    """A synthesized finding and the source numbers referenced in it."""

    text: str
    citation_numbers: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ResearchAnswer:
    """Structured response returned by the research agent."""

    question: str
    summary: str
    findings: tuple[ResearchFinding, ...]
    citations: tuple[ResearchCitation, ...]
    search_provider: str
    llm_provider: str
    llm_model: str
    status: str = "complete"


class ResearchLLMError(RuntimeError):
    """The configured model could not synthesize the retrieved sources."""


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
        ).strip()
    return str(content).strip()


class ResearchAgent:
    """Collect web sources and synthesize a cited answer through AgentX's gateway."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        search_provider: WebSearchProvider | None = None,
        model: ChatModel | None = None,
    ) -> None:
        self._settings = settings
        self._search_provider = search_provider
        self._model = model

    def research(self, question: str, *, max_results: int = 5) -> ResearchAnswer:
        """Search and synthesize sources for a non-empty research question."""
        if not question or not question.strip():
            raise ValueError("Research question must not be empty.")
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")

        settings = self._settings or Settings.from_env()
        provider = self._search_provider or create_web_search_provider(settings)
        try:
            sources = provider.search(question.strip(), max_results=max_results)
        except SearchProviderError:
            raise
        except Exception as exc:
            raise SearchProviderError(
                f"Web search provider '{provider.name}' failed ({type(exc).__name__})."
            ) from exc

        if not sources:
            return ResearchAnswer(
                question=question.strip(),
                summary="No web sources were found for this question.",
                findings=(),
                citations=(),
                search_provider=provider.name,
                llm_provider=settings.llm_provider,
                llm_model=(
                    settings.ollama_model
                    if settings.llm_provider == "ollama"
                    else settings.openai_model
                    if settings.llm_provider == "openai"
                    else settings.gemini_model
                ),
                status="no_sources",
            )

        citations = tuple(
            self._citation(index, source) for index, source in enumerate(sources, 1)
        )
        prompt = self._build_prompt(question.strip(), sources)
        try:
            model = self._model or create_chat_model(settings=settings)
            summary = _response_text(model.invoke(prompt))
            if not summary:
                raise ValueError("The model returned an empty response.")
        except Exception as exc:
            raise ResearchLLMError(
                f"Research synthesis failed using {settings.llm_provider} "
                f"({type(exc).__name__}). Check model connectivity and configuration."
            ) from None

        return ResearchAnswer(
            question=question.strip(),
            summary=summary,
            findings=self._extract_findings(summary, citations),
            citations=citations,
            search_provider=provider.name,
            llm_provider=settings.llm_provider,
            llm_model=(
                settings.ollama_model
                if settings.llm_provider == "ollama"
                else settings.openai_model
                if settings.llm_provider == "openai"
                else settings.gemini_model
            ),
        )

    @staticmethod
    def _citation(index: int, source: WebSource) -> ResearchCitation:
        return ResearchCitation(
            number=index,
            title=source.title,
            url=source.url,
            published_date=source.published_date,
            snippet=source.snippet,
        )

    @staticmethod
    def _build_prompt(question: str, sources: list[WebSource]) -> str:
        formatted_sources = "\n\n".join(
            f"[{index}] {source.title}\nURL: {source.url}\n"
            f"Published: {source.published_date or 'date unavailable'}\n"
            f"Snippet: {source.snippet[:1200]}"
            for index, source in enumerate(sources, 1)
        )
        return (
            "Answer using only the source material below. Write a concise synthesis "
            "with 2-4 factual findings and cite claims inline using [1], [2]. "
            "If sources do not support a claim, say so. Do not invent facts or sources.\n\n"
            f"Research question: {question}\n\nSources:\n{formatted_sources}"
        )

    @staticmethod
    def _extract_findings(
        summary: str, citations: tuple[ResearchCitation, ...]
    ) -> tuple[ResearchFinding, ...]:
        valid_numbers = {citation.number for citation in citations}
        lines = [line.strip() for line in summary.splitlines() if line.strip()]
        if not lines:
            lines = [summary]
        findings: list[ResearchFinding] = []
        for line in lines:
            text = re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", line).strip()
            if not text:
                continue
            referenced = tuple(
                number
                for number in sorted({int(value) for value in _CITATION_PATTERN.findall(text)})
                if number in valid_numbers
            )
            findings.append(ResearchFinding(text=text, citation_numbers=referenced))
        return tuple(findings)
