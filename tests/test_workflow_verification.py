"""Workflow integration tests for the Verification Agent node."""

import unittest
from unittest.mock import Mock

from agentx.agents.rag_agent import RAGAnswer, RAGSource
from agentx.agents.research_agent import ResearchAnswer, ResearchCitation, ResearchFinding
from agentx.agents.verification_agent import VerificationAgent
from agentx.core.workflow import build_workflow


class WorkflowVerificationTests(unittest.TestCase):
    def setUp(self):
        self.research = Mock()
        self.research.research.return_value = ResearchAnswer(
            question="Research recent AI safety findings online.",
            summary="AI systems require evaluation [1].",
            findings=(
                ResearchFinding("AI systems require evaluation [1].", (1,)),
            ),
            citations=(
                ResearchCitation(
                    1,
                    "Web source",
                    "https://example.test",
                    snippet="AI systems require evaluation in safety studies.",
                ),
            ),
            search_provider="mock",
            llm_provider="ollama",
            llm_model="llama3.2:1b",
        )
        self.rag = Mock()
        self.rag.answer.return_value = RAGAnswer(
            question="What does the uploaded PDF say?",
            answer="The report describes model evaluation [1].",
            sources=(
                RAGSource(
                    1,
                    "report.pdf",
                    4,
                    "doc-1",
                    "doc-1:p4:c0001",
                    0.9,
                    0.1,
                    "The report describes model evaluation.",
                ),
            ),
            status="success",
        )
        self.verifier = Mock(wraps=VerificationAgent())

    def make_workflow(self):
        return build_workflow(
            research_agent=self.research,
            rag_agent=self.rag,
            verification_agent=self.verifier,
        )

    def test_research_only_path_runs_verifier_and_preserves_research_outputs(self):
        result = self.make_workflow().invoke(
            {"user_query": "Research recent AI safety findings online."}
        )

        self.assertEqual(result["status"], "research_complete")
        self.assertEqual(result["verification_status"], "supported")
        self.assertEqual(result["research_citations"][0]["title"], "Web source")
        self.assertTrue(result["verification_results"])
        self.verifier.verify.assert_called_once()

    def test_document_only_path_runs_verifier_and_preserves_rag_outputs(self):
        result = self.make_workflow().invoke(
            {"user_query": "What does the uploaded PDF say about evaluation?"}
        )

        self.assertEqual(result["status"], "knowledge_complete")
        self.assertEqual(result["verification_status"], "supported")
        self.assertEqual(result["rag_answer"], "The report describes model evaluation [1].")
        self.assertEqual(result["retrieved_context"][0]["chunk_id"], "doc-1:p4:c0001")
        self.verifier.verify.assert_called_once()

    def test_combined_path_verifies_both_agent_results(self):
        result = self.make_workflow().invoke(
            {
                "user_query": "Compare this report with current web research.",
                "document_id": "doc-1",
            }
        )

        self.assertEqual(result["status"], "combined_complete")
        self.assertEqual(result["verification_status"], "supported")
        self.assertTrue(result["research_results"])
        self.assertTrue(result["retrieved_context"])
        self.assertEqual(len(result["verification_results"]), 2)
        self.verifier.verify.assert_called_once()

    def test_partial_combined_path_keeps_successful_evidence_and_verifies_it(self):
        self.rag.answer.side_effect = RuntimeError("PDF store unavailable")
        result = self.make_workflow().invoke(
            {
                "user_query": "Compare this report with current web research.",
                "document_id": "doc-1",
            }
        )

        self.assertEqual(result["status"], "combined_partial_success")
        self.assertEqual(result["verification_status"], "supported")
        self.assertEqual(result["source_evidence"][0]["source_type"], "web")
        self.assertTrue(result["errors"])

    def test_verification_error_is_recorded_without_overwriting_agent_status(self):
        self.verifier.verify.side_effect = RuntimeError("checker failed")
        result = self.make_workflow().invoke(
            {"user_query": "Research recent AI safety findings online."}
        )

        self.assertEqual(result["status"], "research_complete")
        self.assertEqual(result["verification_status"], "error")
        self.assertIn("checker failed", result["errors"][-1])
        self.assertTrue(result["research_results"])

    def test_clarification_route_stays_unchanged_and_skips_verification(self):
        result = self.make_workflow().invoke({"user_query": "Can you help me?"})

        self.assertEqual(result["route"], "clarification")
        self.assertEqual(result["status"], "clarification_required")
        self.assertFalse(self.verifier.verify.called)


if __name__ == "__main__":
    unittest.main()
