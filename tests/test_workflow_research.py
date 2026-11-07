"""Tests connecting the existing Research Agent to the LangGraph workflow."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agentx.agents import ResearchAgent
from agentx.core.settings import Settings
from agentx.core.workflow import build_workflow
from agentx.tools.web_search import WebSource


class WorkflowResearchTests(unittest.TestCase):
    def setUp(self):
        self.search = Mock()
        self.search.name = "mock-search"
        self.search.search.return_value = [
            WebSource(
                title="Research source",
                url="https://example.test/research",
                snippet="Evidence from the source.",
                published_date="2026-01-02",
            )
        ]
        self.model = Mock()
        self.model.invoke.return_value = SimpleNamespace(
            content="The evidence supports this finding [1]."
        )
        self.agent = ResearchAgent(
            settings=Settings(), search_provider=self.search, model=self.model
        )

    def test_research_agent_runs_through_graph_and_saves_findings_and_citations(self):
        result = build_workflow(research_agent=self.agent).invoke(
            {"user_query": "Research recent evidence about this topic."}
        )

        self.assertEqual(result["route"], "research")
        self.assertEqual(result["status"], "research_complete")
        self.search.search.assert_called_once_with(
            "Research recent evidence about this topic.", max_results=5
        )
        finding = result["research_results"][0]
        self.assertEqual(finding["text"], "The evidence supports this finding [1].")
        self.assertEqual(finding["citation_numbers"], (1,))
        citation = result["research_citations"][0]
        self.assertEqual(citation["title"], "Research source")
        self.assertEqual(citation["url"], "https://example.test/research")
        self.assertEqual(citation["published_date"], "2026-01-02")
        self.assertEqual(citation["snippet"], "Evidence from the source.")
        evidence = result["source_evidence"][0]
        self.assertEqual(evidence["source_type"], "web")
        self.assertEqual(evidence["source_reference"], "https://example.test/research")
        self.assertEqual(evidence["text"], "Evidence from the source.")
        self.assertEqual(evidence["citation_id"], 1)
        self.assertEqual(result["research_summary"], "The evidence supports this finding [1].")

    def test_user_query_is_passed_to_existing_research_agent(self):
        agent = Mock()
        agent.research.return_value = SimpleNamespace(
            status="complete", summary="done", findings=(), citations=()
        )
        query = "Search online for current evidence."

        build_workflow(research_agent=agent).invoke({"user_query": query})

        agent.research.assert_called_once_with(query)

    def test_empty_search_results_are_preserved_without_model_call(self):
        self.search.search.return_value = []
        result = build_workflow(research_agent=self.agent).invoke(
            {"user_query": "Find current sources about this topic."}
        )

        self.assertEqual(result["status"], "research_no_sources")
        self.assertEqual(result["research_results"], [])
        self.assertEqual(result["research_citations"], [])
        self.assertEqual(result["research_summary"], "No web sources were found for this question.")
        self.model.invoke.assert_not_called()

    def test_research_agent_failure_is_recorded_in_state(self):
        agent = Mock()
        agent.research.side_effect = RuntimeError("provider unavailable")

        result = build_workflow(research_agent=agent).invoke(
            {"user_query": "Research current reports online."}
        )

        self.assertEqual(result["route"], "research")
        self.assertEqual(result["status"], "research_error")
        self.assertIn("provider unavailable", result["errors"][-1])


if __name__ == "__main__":
    unittest.main()
