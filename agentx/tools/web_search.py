"""Provider-neutral web search abstraction with demo and Tavily backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agentx.core.settings import ConfigurationError, Settings


@dataclass(frozen=True, slots=True)
class WebSource:
    """A normalized web search result."""

    title: str
    url: str
    snippet: str
    published_date: str | None = None


class WebSearchProvider(Protocol):
    """Search interface consumed by research agents."""

    name: str

    def search(self, query: str, *, max_results: int = 5) -> list[WebSource]:
        """Return normalized sources relevant to a query."""
        ...


class SearchConfigurationError(ConfigurationError):
    """The selected search provider is missing required configuration."""


class SearchProviderError(RuntimeError):
    """A configured search provider failed to return results."""


class DemoSearchProvider:
    """Deterministic sample data; its example.com sources are not live research."""

    name = "demo"
    _sources = (
        WebSource(
            title="Example Research Source: AI assistants",
            url="https://example.com/demo/ai-assistants",
            snippet=(
                "Demo fixture: AI research assistants can help organize sources, "
                "summarize material, and support evidence-backed research workflows."
            ),
            published_date="2025-01-15",
        ),
        WebSource(
            title="Example Research Source: Evidence and citations",
            url="https://example.com/demo/evidence-and-citations",
            snippet=(
                "Demo fixture: research systems should distinguish source evidence "
                "from generated summaries and preserve links for verification."
            ),
        ),
    )

    def search(self, query: str, *, max_results: int = 5) -> list[WebSource]:
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")
        return list(self._sources[:max_results])


class TavilySearchProvider:
    """Tavily API adapter implemented with the existing HTTPX dependency."""

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, *, timeout_seconds: float = 15.0) -> None:
        if not api_key:
            raise SearchConfigurationError(
                "TAVILY_API_KEY is required when WEB_SEARCH_PROVIDER is 'tavily'."
            )
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def search(self, query: str, *, max_results: int = 5) -> list[WebSource]:
        if not query.strip():
            raise ValueError("Search query must not be empty.")
        if max_results < 1:
            raise ValueError("max_results must be at least 1.")
        try:
            import httpx
        except ImportError as exc:
            raise SearchProviderError(
                "HTTPX is required for Tavily search; install requirements.txt."
            ) from exc

        payload = {
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "include_published_date": True,
        }
        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(
                    self.endpoint,
                    json=payload,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            raise SearchProviderError(
                f"Tavily search returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.RequestError as exc:
            raise SearchProviderError(
                f"Tavily request failed ({type(exc).__name__})."
            ) from exc

        sources: list[WebSource] = []
        for item in data.get("results", []):
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            snippet = str(item.get("content") or item.get("snippet") or "").strip()
            if not title or not url:
                continue
            published = item.get("published_date") or item.get("published_at")
            sources.append(
                WebSource(
                    title=title,
                    url=url,
                    snippet=snippet,
                    published_date=str(published) if published else None,
                )
            )
        return sources


def create_web_search_provider(settings: Settings | None = None) -> WebSearchProvider:
    """Build the provider configured in settings without making a request."""
    try:
        configured = settings or Settings.from_env()
        configured.validate()
    except ConfigurationError as exc:
        if "WEB_SEARCH_PROVIDER" in str(exc) or "TAVILY_API_KEY" in str(exc):
            raise SearchConfigurationError(str(exc)) from exc
        raise
    if configured.web_search_provider == "demo":
        return DemoSearchProvider()
    if configured.web_search_provider == "tavily":
        return TavilySearchProvider(configured.tavily_api_key or "")
    raise SearchConfigurationError(
        f"Unsupported WEB_SEARCH_PROVIDER {configured.web_search_provider!r}; "
        "choose 'demo' or 'tavily'."
    )
