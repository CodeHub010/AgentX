"""Tests for deterministic evidence-aware Markdown report assembly."""

import unittest

from agentx.agents.report_generator import ReportGenerator


class ReportGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = ReportGenerator()

    def test_research_only_report_preserves_real_web_reference(self):
        result = self.generator.generate({
            "user_query": "What does the study find?",
            "research_summary": "CNN detects threats [1].",
            "research_results": [{"text": "CNN detects threats [1].", "citation_numbers": [1]}],
            "source_evidence": [{
                "source_id": "web:1", "source_type": "web", "citation_id": 1,
                "source_title": "Threat study", "source_reference": "https://example.test/study",
                "text": "CNN detects threats in network traffic.",
            }],
            "verification_results": [{"claim": "CNN detects threats [1].", "status": "supported", "supporting_sources": [], "explanation": "Lexical overlap."}],
        })
        self.assertEqual(result.status, "generated")
        self.assertIn("## Web Research Findings", result.markdown)
        self.assertIn("Web [1]", result.markdown)
        self.assertIn("https://example.test/study", result.markdown)
        self.assertIn("Threat study", result.markdown)

    def test_document_report_preserves_passage_and_metadata(self):
        result = self.generator.generate({
            "user_query": "What does the PDF say?", "rag_answer": "CNN detects threats [1].",
            "source_evidence": [{
                "source_id": "document:d1:c1", "source_type": "document", "citation_id": 1,
                "filename": "report.pdf", "source_title": "report.pdf", "page_number": 8,
                "document_id": "d1", "chunk_id": "d1:p8:c1", "similarity": 0.91,
                "text": "CNN detects threats in network traffic.",
            }],
            "verification_results": [{"claim": "CNN detects threats [1].", "status": "supported", "supporting_sources": [], "explanation": "Lexical overlap."}],
        })
        self.assertIn("report.pdf, page 8", result.markdown)
        self.assertIn("document id d1", result.markdown)
        self.assertIn("chunk id d1:p8:c1", result.markdown)
        self.assertIn("> CNN detects threats in network traffic.", result.markdown)
        self.assertIn("similarity 0.91", result.markdown)

    def test_multi_claim_document_answer_uses_claim_level_verification_status(self):
        result = self.generator.generate({
            "rag_answer": "| Filters: 64 [1] |\n| Kernel Size: 3 [2] |",
            "verification_status": "supported",
            "verification_results": [
                {"claim": "| Filters: 64 [1] |", "status": "supported"},
                {"claim": "| Kernel Size: 3 [2] |", "status": "supported"},
            ],
            "source_evidence": [],
        })
        self.assertIn("Document QA response (verification status: supported)", result.markdown)
        self.assertNotIn("verification unavailable", result.markdown)

    def test_document_answer_status_summarizes_concerns_before_supported(self):
        result = self.generator.generate({
            "rag_answer": "Filters: 64. Kernel Size: 3.",
            "verification_status": "insufficient_evidence",
            "verification_results": [
                {"claim": "Filters: 64.", "status": "supported"},
                {"claim": "Kernel Size: 3.", "status": "insufficient_evidence"},
            ],
        })
        self.assertIn("Document QA response (verification status: insufficient_evidence)", result.markdown)

    def test_combined_report_preserves_both_sources_and_verification_statuses(self):
        result = self.generator.generate({
            "user_query": "Compare web and PDF results",
            "research_results": [{"text": "The model detects 92% [1].", "citation_numbers": [1]}],
            "rag_answer": "The model detects 81% [1].",
            "source_evidence": [
                {"source_id": "web:1", "source_type": "web", "citation_id": 1, "source_reference": "https://example.test", "text": "The model detects 92%."},
                {"source_id": "doc:d:c", "source_type": "document", "citation_id": 1, "filename": "report.pdf", "page_number": 3, "text": "The model detects 81%."},
            ],
            "verification_results": [
                {"claim": "The model detects 92% [1].", "status": "conflicting", "supporting_sources": [{"source_type": "web", "number": 1}, {"source_type": "document", "number": 1}], "explanation": "Different values.", "contradiction": "92 vs 81"},
                {"claim": "The model detects 81% [1].", "status": "unsupported", "supporting_sources": [], "explanation": "Not supported."},
            ],
        })
        self.assertIn("https://example.test", result.markdown)
        self.assertIn("report.pdf, page 3", result.markdown)
        self.assertIn("**conflicting**", result.markdown)
        self.assertIn("**unsupported**", result.markdown)
        self.assertIn("92 vs 81", result.markdown)
        self.assertIn("Conflicting or Unsupported Claims", result.markdown)

    def test_missing_text_and_insufficient_evidence_are_warned_about(self):
        result = self.generator.generate({
            "user_query": "Question?", "research_citations": [{"number": 1}],
            "source_evidence": [{"source_id": "web:1", "source_type": "web", "citation_id": 1, "source_reference": "https://example.test"}],
            "verification_results": [{"claim": "A claim", "status": "insufficient_evidence", "explanation": "Text missing."}],
        })
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIn("no source text is available", result.markdown)
        self.assertIn("A claim: Text missing.", result.markdown)

    def test_does_not_invent_citations_or_source_content(self):
        result = self.generator.generate({
            "user_query": "Question?", "research_results": [{"text": "Claim [9].", "citation_numbers": [9]}],
            "source_evidence": [{"source_id": "web:1", "source_type": "web", "citation_id": 1, "text": "Actual passage."}],
        })
        self.assertIn("No validated source reference is available.", result.markdown)
        self.assertNotIn("Claim [9]. References: Web [9]", result.markdown)
        self.assertNotIn("Citation [9]", result.markdown)

    def test_empty_or_missing_state_still_produces_a_clear_report(self):
        result = self.generator.generate({})
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIn("# AgentX Research Report", result.markdown)
        self.assertIn("No research question was supplied.", result.markdown)
        self.assertIn("No source references are available.", result.markdown)


if __name__ == "__main__":
    unittest.main()
