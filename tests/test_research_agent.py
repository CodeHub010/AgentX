"""Unit tests for the research agent and web search providers."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agentx.agents import ResearchAgent
from agentx.agents.research_agent import ResearchLLMError
from agentx.core.settings import ConfigurationError, Settings
from agentx.tools.web_search import (
    DemoSearchProvider,
    SearchConfigurationError,
    SearchProviderError,
    TavilySearchProvider,
    WebSource,
    create_web_search_provider,
)


class WebSearchTests(unittest.TestCase):
    def test_demo_provider_returns_labeled_fixture_data(self) -> None:
        results = DemoSearchProvider().search("AI research assistants")
        self.assertTrue(results)
        self.assertTrue(all("example.com/demo/" in item.url for item in results))
        self.assertEqual(results[0].published_date, "2025-01-15")

    def test_tavily_requires_api_key(self) -> None:
        with self.assertRaisesRegex(SearchConfigurationError, "TAVILY_API_KEY is required"):
            create_web_search_provider(
                Settings(web_search_provider="tavily", tavily_api_key=None)
            )

    def test_unknown_search_provider_is_rejected(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "Unsupported WEB_SEARCH_PROVIDER"):
            Settings.from_env({"WEB_SEARCH_PROVIDER": "unknown"})

    def test_tavily_maps_search_result_fields_using_mock_http(self) -> None:
        response = Mock()
        response.json.return_value = {
            "results": [
                {
                    "title": "A source",
                    "url": "https://research.example/source",
                    "content": "Relevant snippet",
                    "published_date": "2025-02-03",
                }
            ]
        }
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.post.return_value = response
        httpx_stub = types.SimpleNamespace(Client=Mock(return_value=client))

        with patch.dict(sys.modules, {"httpx": httpx_stub}):
            source = TavilySearchProvider("unit-test-key").search("query")[0]

        self.assertEqual(source.title, "A source")
        self.assertEqual(source.url, "https://research.example/source")
        self.assertEqual(source.snippet, "Relevant snippet")
        self.assertEqual(source.published_date, "2025-02-03")
        self.assertEqual(
            client.post.call_args.kwargs["headers"],
            {"Authorization": "Bearer unit-test-key"},
        )


class ResearchAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = [
            WebSource(
                title="Source One",
                url="https://example.org/one",
                snippet="First evidence snippet.",
                published_date="2024-05-01",
            ),
            WebSource(
                title="Source Two",
                url="https://example.org/two",
                snippet="Second evidence snippet.",
            ),
        ]
        self.search = Mock()
        self.search.name = "mock"
        self.search.search.return_value = self.sources
        self.model = Mock()
        self.model.invoke.return_value = SimpleNamespace(
            content="- Evidence one supports the claim [1].\n- Evidence two adds context [2]."
        )
        self.settings = Settings(ollama_model="llama3.2:1b")

    def test_returns_structured_findings_and_citations(self) -> None:
        result = ResearchAgent(
            settings=self.settings, search_provider=self.search, model=self.model
        ).research("What do the sources say?")

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.search_provider, "mock")
        self.assertEqual(result.llm_model, "llama3.2:1b")
        self.assertEqual([citation.number for citation in result.citations], [1, 2])
        self.assertEqual(result.citations[0].published_date, "2024-05-01")
        self.assertEqual(result.citations[0].snippet, "First evidence snippet.")
        self.assertEqual(result.findings[0].text, "Evidence one supports the claim [1].")
        self.assertEqual(result.findings[0].citation_numbers, (1,))
        prompt = self.model.invoke.call_args.args[0]
        self.assertIn("[1] Source One", prompt)
        self.assertIn("First evidence snippet.", prompt)

    def test_empty_search_results_return_status_without_calling_model(self) -> None:
        self.search.search.return_value = []
        result = ResearchAgent(
            settings=self.settings, search_provider=self.search, model=self.model
        ).research("A question")
        self.assertEqual(result.status, "no_sources")
        self.assertEqual(result.citations, ())
        self.model.invoke.assert_not_called()

    def test_llm_failure_is_reported_as_research_error(self) -> None:
        self.model.invoke.side_effect = TimeoutError("model timeout")
        with self.assertRaisesRegex(ResearchLLMError, "Research synthesis failed"):
            ResearchAgent(
                settings=self.settings, search_provider=self.search, model=self.model
            ).research("A question")

    def test_empty_question_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ResearchAgent(settings=self.settings, search_provider=self.search).research(" ")

    def test_search_failure_is_reported(self) -> None:
        self.search.search.side_effect = OSError("provider offline")
        with self.assertRaisesRegex(SearchProviderError, "provider 'mock' failed"):
            ResearchAgent(
                settings=self.settings, search_provider=self.search, model=self.model
            ).research("A question")


if __name__ == "__main__":
    unittest.main()
