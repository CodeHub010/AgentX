"""Tests for the combined web research and document QA workflow route."""

import unittest
from unittest.mock import Mock

from agentx.agents.rag_agent import RAGAnswer, RAGSource
from agentx.agents.research_agent import ResearchAnswer, ResearchCitation, ResearchFinding
from agentx.core.workflow import build_workflow


def make_research_result(*, status="complete", summary="Web summary"):
    return ResearchAnswer(
        question="Combined question",
        summary=summary,
        findings=(ResearchFinding(text="Web finding [1].", citation_numbers=(1,)),)
        if status == "complete"
        else (),
        citations=(
            ResearchCitation(
                number=1,
                title="Web source",
                url="https://example.test/source",
                published_date="2026-02-03",
                snippet="Web evidence supports the finding.",
            ),
        )
        if status == "complete"
        else (),
        search_provider="mock",
        llm_provider="ollama",
        llm_model="llama3.2:1b",
        status=status,
    )


def make_rag_result(*, status="success", answer="Document finding [1]."):
    sources = (
        RAGSource(
            number=1,
            filename="report.pdf",
            page_number=9,
            document_id="doc-789",
            chunk_id="doc-789:p9:c0001",
            similarity=0.91,
            distance=0.09,
            text="Document evidence supports the answer.",
        ),
    ) if status == "success" else ()
    return RAGAnswer(
        question="Combined question",
        answer=answer,
        sources=sources,
        status=status,
        error_message="RAG service unavailable" if status == "error" else None,
    )


class CombinedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.research = Mock()
        self.research.research.return_value = make_research_result()
        self.rag = Mock()
        self.rag.answer.return_value = make_rag_result()
        self.query = "Compare this report with current web research."
        self.state = {"user_query": self.query, "document_id": "doc-789"}

    def run_combined(self):
        return build_workflow(
            research_agent=self.research,
            rag_agent=self.rag,
        ).invoke(self.state)

    def test_both_agents_succeed_and_preserve_findings_citations_and_sources(self):
        result = self.run_combined()

        self.assertEqual(result["route"], "combined")
        self.assertEqual(result["status"], "combined_complete")
        self.assertEqual(result["research_results"][0]["text"], "Web finding [1].")
        self.assertEqual(result["research_citations"][0]["url"], "https://example.test/source")
        self.assertEqual(result["rag_answer"], "Document finding [1].")
        self.assertEqual(result["retrieved_context"][0]["filename"], "report.pdf")
        self.assertEqual(result["retrieved_context"][0]["document_id"], "doc-789")
        self.assertEqual(result["retrieved_context"][0]["chunk_id"], "doc-789:p9:c0001")
        self.assertEqual(result["rag_citations"][0]["text"], "[1] report.pdf, page 9")
        self.assertEqual(
            [item["source_type"] for item in result["source_evidence"]],
            ["web", "document"],
        )
        self.assertEqual(
            result["source_evidence"][0]["text"],
            "Web evidence supports the finding.",
        )
        self.assertEqual(
            result["source_evidence"][1]["text"],
            "Document evidence supports the answer.",
        )
        self.assertEqual(
            len({item["source_id"] for item in result["source_evidence"]}), 2
        )

    def test_same_query_and_document_id_are_forwarded_to_both_agents(self):
        self.run_combined()

        self.research.research.assert_called_once_with(self.query)
        self.rag.answer.assert_called_once_with(self.query, document_id="doc-789")

    def test_research_succeeds_and_rag_fails_without_losing_research(self):
        self.rag.answer.return_value = make_rag_result(
            status="error", answer="Could not retrieve PDF evidence."
        )

        result = self.run_combined()

        self.assertEqual(result["status"], "combined_partial_success")
        self.assertEqual(result["research_results"][0]["text"], "Web finding [1].")
        self.assertIn("RAG service unavailable", result["errors"][-1])
        self.assertEqual(result["rag_answer"], "Could not retrieve PDF evidence.")
        self.assertEqual(len(result["source_evidence"]), 1)
        self.assertEqual(result["source_evidence"][0]["source_type"], "web")

    def test_rag_succeeds_and_research_fails_without_losing_rag_result(self):
        self.research.research.side_effect = RuntimeError("search unavailable")

        result = self.run_combined()

        self.assertEqual(result["status"], "combined_partial_success")
        self.assertEqual(result["rag_answer"], "Document finding [1].")
        self.assertEqual(result["retrieved_context"][0]["document_id"], "doc-789")
        self.assertIn("search unavailable", result["errors"][-1])
        self.assertNotIn("research_results", result)
        self.assertEqual(len(result["source_evidence"]), 1)
        self.assertEqual(result["source_evidence"][0]["source_type"], "document")

    def test_both_agents_fail_returns_total_failure_and_both_errors(self):
        self.research.research.side_effect = RuntimeError("search unavailable")
        self.rag.answer.return_value = make_rag_result(
            status="error", answer="Could not retrieve PDF evidence."
        )

        result = self.run_combined()

        self.assertEqual(result["status"], "combined_failed")
        self.assertEqual(len(result["errors"]), 2)
        self.assertIn("search unavailable", result["errors"][0])
        self.assertIn("RAG service unavailable", result["errors"][1])

    def test_both_agents_returning_no_evidence_sets_insufficient_status(self):
        self.research.research.return_value = make_research_result(
            status="no_sources", summary="No web sources were found."
        )
        self.rag.answer.return_value = make_rag_result(
            status="insufficient_evidence",
            answer="No relevant PDF passage was found.",
        )

        result = self.run_combined()

        self.assertEqual(result["status"], "combined_insufficient_evidence")
        self.assertEqual(result["research_results"], [])
        self.assertEqual(result["research_citations"], [])
        self.assertEqual(result["rag_answer"], "No relevant PDF passage was found.")
        self.assertEqual(result["retrieved_context"], [])

    def test_existing_individual_routes_still_use_their_agent(self):
        research_state = build_workflow(
            research_agent=self.research, rag_agent=self.rag
        ).invoke({"user_query": "Research recent results online."})
        knowledge_state = build_workflow(
            research_agent=self.research, rag_agent=self.rag
        ).invoke({"user_query": "What does the uploaded report say?"})

        self.assertEqual(research_state["route"], "research")
        self.assertEqual(research_state["status"], "research_complete")
        self.assertEqual(knowledge_state["route"], "knowledge")
        self.assertEqual(knowledge_state["status"], "knowledge_complete")


if __name__ == "__main__":
    unittest.main()
