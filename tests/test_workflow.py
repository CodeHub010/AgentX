"""Focused tests for the initial AgentX LangGraph workflow."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

from agentx.core.workflow import build_workflow


class WorkflowTests(unittest.TestCase):
    def test_graph_builds(self):
        workflow = build_workflow()
        self.assertIsNotNone(workflow)

    def test_each_classified_query_reaches_its_route(self):
        cases = (
            ("Research recent AI safety news online.", "web_research", "research"),
            ("What does the uploaded PDF say?", "document_qa", "knowledge"),
            (
                "Compare this report with current web research.",
                "combined_research",
                "combined",
            ),
            ("Can you help me with this?", "unknown", "clarification"),
        )
        for query, task_type, route in cases:
            with self.subTest(task_type=task_type):
                research_agent = Mock()
                research_agent.research.return_value = SimpleNamespace(
                    status="complete", summary="done", findings=(), citations=()
                )
                rag_agent = Mock()
                rag_agent.answer.return_value = SimpleNamespace(
                    status="success", answer="done", sources=(), error_message=None
                )
                result = build_workflow(
                    research_agent=research_agent, rag_agent=rag_agent
                ).invoke({"user_query": query})
                self.assertEqual(result["user_query"], query)
                self.assertEqual(result["task_type"], task_type)
                self.assertEqual(result["route"], route)
                expected_status = {
                    "research": "research_complete",
                    "knowledge": "knowledge_complete",
                    "combined": "combined_complete",
                    "clarification": "clarification_required",
                }[route]
                self.assertEqual(result["status"], expected_status)

    def test_empty_query_has_safe_unknown_classification(self):
        result = build_workflow().invoke({"user_query": " "})
        self.assertEqual(result["task_type"], "unknown")
        self.assertEqual(result["route"], "clarification")
        self.assertEqual(result["status"], "clarification_required")
        self.assertTrue(result["errors"])

    def test_missing_or_malformed_query_routes_to_clarification(self):
        for state in ({}, {"user_query": 123}):
            with self.subTest(state=state):
                result = build_workflow().invoke(state)
                self.assertEqual(result["task_type"], "unknown")
                self.assertEqual(result["route"], "clarification")
                self.assertEqual(result["status"], "clarification_required")

    def test_classification_failure_is_recorded_safely(self):
        with patch("agentx.core.workflow.classify_task", side_effect=RuntimeError):
            result = build_workflow().invoke({"user_query": "Research AI."})

        self.assertEqual(result["task_type"], "unknown")
        self.assertEqual(result["route"], "clarification")
        self.assertEqual(result["status"], "clarification_required")
        self.assertIn("Task classification failed", result["errors"][0])


if __name__ == "__main__":
    unittest.main()
