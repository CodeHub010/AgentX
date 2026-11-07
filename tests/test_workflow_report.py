"""Workflow tests for post-verification report generation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agentx.agents.rag_agent import RAGSource
from agentx.agents.research_agent import ResearchCitation, ResearchFinding
from agentx.core.workflow import build_workflow


class WorkflowReportTests(unittest.TestCase):
    def _agents(self):
        research = Mock()
        research.research.return_value = SimpleNamespace(
            status="complete", summary="CNN detects threats [1].",
            findings=(ResearchFinding(text="CNN detects threats [1].", citation_numbers=(1,)),),
            citations=(ResearchCitation(number=1, title="Study", url="https://example.test", published_date=None, snippet="CNN detects threats."),),
        )
        rag = Mock()
        rag.answer.return_value = SimpleNamespace(
            status="success", answer="CNN detects threats [1].", error_message=None,
            sources=(RAGSource(number=1, filename="report.pdf", page_number=4, document_id="d1", chunk_id="d1:p4:c1", similarity=0.9, distance=0.1, text="CNN detects threats."),),
        )
        return research, rag

    def test_research_route_generates_report_after_verification(self):
        research, rag = self._agents()
        result = build_workflow(research_agent=research, rag_agent=rag).invoke(
            {"user_query": "Research CNN threat detection online."}
        )
        self.assertIn("## Evidence Verification", result["final_report"])
        self.assertIn("https://example.test", result["final_report"])
        self.assertEqual(result["report_status"], "generated")
        self.assertEqual(result["verification_status"], "supported")
        rag.answer.assert_not_called()

    def test_document_route_generates_report(self):
        research, rag = self._agents()
        result = build_workflow(research_agent=research, rag_agent=rag).invoke(
            {"user_query": "What does the uploaded PDF say?", "document_id": "d1"}
        )
        self.assertIn("report.pdf, page 4", result["final_report"])
        self.assertEqual(result["report_status"], "generated")
        research.research.assert_not_called()

    def test_combined_route_generates_report_with_both_evidence_types(self):
        research, rag = self._agents()
        result = build_workflow(research_agent=research, rag_agent=rag).invoke(
            {"user_query": "Compare the uploaded report with current web research."}
        )
        self.assertIn("Web citation [1]", result["final_report"])
        self.assertIn("Document citation [1]", result["final_report"])
        self.assertEqual(result["report_status"], "generated")

    def test_report_generation_failure_is_recorded(self):
        research, rag = self._agents()
        generator = Mock()
        generator.generate.side_effect = RuntimeError("assembly error")
        result = build_workflow(
            research_agent=research, rag_agent=rag, report_generator=generator
        ).invoke({"user_query": "Research CNN threat detection online."})
        self.assertEqual(result["report_status"], "error")
        self.assertTrue(any("Report generation failed" in error for error in result["errors"]))

    def test_clarification_route_does_not_run_report_generation(self):
        generator = Mock()
        result = build_workflow(report_generator=generator).invoke(
            {"user_query": "Can you help me with this?"}
        )
        self.assertEqual(result["route"], "clarification")
        self.assertEqual(result["status"], "clarification_required")
        self.assertNotIn("final_report", result)
        generator.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
