"""Tests for transparent, deterministic evidence verification."""

import unittest

from agentx.agents.verification_agent import VerificationAgent


class VerificationAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = VerificationAgent()

    def test_supported_claim_requires_matching_citation_and_source_text(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {"text": "CNN detects insider threats [1].", "citation_numbers": [1]}
                ],
                "research_citations": [
                    {
                        "number": 1,
                        "title": "Study",
                        "url": "https://example.test/study",
                        "snippet": "CNN detects insider threats in network traffic.",
                    }
                ],
            }
        )

        self.assertEqual(report.findings[0].status, "supported")
        self.assertEqual(report.findings[0].supporting_sources[0]["title"], "Study")
        self.assertIn("not a guarantee of entailment", report.findings[0].explanation)

    def test_verification_uses_preserved_source_evidence(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {"text": "CNN detects insider threats [1].", "citation_numbers": [1]}
                ],
                "research_citations": [{"number": 1, "title": "Study"}],
                "source_evidence": [
                    {
                        "source_id": "web:1",
                        "source_type": "web",
                        "source_reference": "https://example.test/study",
                        "source_title": "Study",
                        "text": "CNN detects insider threats in network traffic.",
                        "citation_id": 1,
                    }
                ],
            }
        )
        self.assertEqual(report.findings[0].status, "supported")
        self.assertEqual(
            report.findings[0].supporting_sources[0]["source_id"], "web:1"
        )

    def test_claim_without_citation_is_insufficient_not_a_contradiction(self):
        report = self.agent.verify(
            {"research_results": [{"text": "CNN detects insider threats."}]}
        )
        self.assertEqual(report.findings[0].status, "insufficient_evidence")
        self.assertIn("no citation", report.findings[0].explanation)

    def test_invalid_citation_reference_is_unsupported(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {"text": "CNN detects insider threats [2].", "citation_numbers": [2]}
                ],
                "research_citations": [{"number": 1, "title": "Available source"}],
            }
        )
        self.assertEqual(report.findings[0].status, "unsupported")
        self.assertIn("[2]", report.findings[0].explanation)

    def test_citation_metadata_without_source_text_is_insufficient(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {"text": "CNN detects insider threats [1].", "citation_numbers": [1]}
                ],
                "research_citations": [{"number": 1, "title": "Study"}],
            }
        )
        self.assertEqual(report.findings[0].status, "insufficient_evidence")
        self.assertIn("existence alone", report.findings[0].explanation)

    def test_available_source_text_that_does_not_support_claim_is_unsupported(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {"text": "CNN detects insider threats [1].", "citation_numbers": [1]}
                ],
                "research_citations": [
                    {"number": 1, "snippet": "Weather forecasts use satellite images."}
                ],
            }
        )
        self.assertEqual(report.findings[0].status, "unsupported")

    def test_conflicting_web_and_pdf_evidence_is_reported(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {
                        "text": "The model detects 92% of threats [1].",
                        "citation_numbers": [1],
                    }
                ],
                "research_citations": [
                    {"number": 1, "title": "Web study", "snippet": "Model detects 92% of threats."}
                ],
                "retrieved_context": [
                    {
                        "number": 1,
                        "filename": "report.pdf",
                        "page_number": 7,
                        "document_id": "doc-1",
                        "chunk_id": "doc-1:p7:c0001",
                        "text": "The model detects 81% of threats.",
                    }
                ],
            }
        )

        result = report.findings[0]
        self.assertEqual(result.status, "conflicting")
        self.assertIn("different numeric values", result.contradiction)
        self.assertEqual(
            {source["source_type"] for source in result.supporting_sources},
            {"web", "document"},
        )

    def test_research_only_request_is_checked(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {"text": "CNN detects threats [1].", "citation_numbers": [1]}
                ],
                "research_citations": [
                    {"number": 1, "snippet": "CNN detects threats."}
                ],
            }
        )
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].status, "supported")

    def test_document_only_request_is_checked(self):
        report = self.agent.verify(
            {
                "rag_answer": "CNN detects threats [1].",
                "retrieved_context": [
                    {
                        "number": 1,
                        "filename": "report.pdf",
                        "page_number": 8,
                        "text": "CNN detects threats in network traffic.",
                    }
                ],
            }
        )
        self.assertEqual(report.findings[0].status, "supported")
        self.assertEqual(
            report.findings[0].supporting_sources[0]["filename"], "report.pdf"
        )

    def test_combined_research_checks_web_and_pdf_claims(self):
        report = self.agent.verify(
            {
                "research_results": [
                    {"text": "CNN detects threats [1].", "citation_numbers": [1]}
                ],
                "research_citations": [
                    {"number": 1, "title": "Web source", "snippet": "CNN detects threats."}
                ],
                "rag_answer": "LSTM detects sequences [1].",
                "retrieved_context": [
                    {
                        "number": 1,
                        "filename": "report.pdf",
                        "page_number": 8,
                        "text": "LSTM detects sequences.",
                    }
                ],
            }
        )
        self.assertEqual([item.status for item in report.findings], ["supported", "supported"])

    def test_no_evidence_and_malformed_state_are_handled_safely(self):
        for state in ({}, {"research_results": "bad data", "retrieved_context": None}):
            with self.subTest(state=state):
                report = self.agent.verify(state)
                self.assertEqual(report.status, "insufficient_evidence")
                self.assertEqual(report.findings[0].status, "insufficient_evidence")

    def test_rag_insufficient_evidence_response_is_not_mislabeled_unsupported(self):
        report = self.agent.verify(
            {
                "rag_answer": "I could not find relevant PDF passages.",
                "rag_status": "insufficient_evidence",
                "retrieved_context": [],
            }
        )
        self.assertEqual(report.findings[0].status, "insufficient_evidence")

    def test_paraphrased_passage_is_recognized_with_conservative_aliases(self):
        report = self.agent.verify({
            "research_results": [{
                "text": "CNN extracts spatial features [1].", "citation_numbers": [1]
            }],
            "research_citations": [{
                "number": 1,
                "snippet": "Convolutional neural networks capture spatial patterns.",
            }],
        })
        self.assertEqual(report.findings[0].status, "supported")
        self.assertIn("wording normalization", report.findings[0].explanation)

    def test_unrelated_sentence_negation_does_not_create_false_conflict(self):
        report = self.agent.verify({
            "rag_answer": "CNN detects threats [1].",
            "retrieved_context": [{
                "number": 1,
                "filename": "report.pdf",
                "page_number": 3,
                "chunk_id": "doc:p3:c1",
                "text": "CNN detects threats in network traffic. Weather does not predict storms.",
            }],
        })
        self.assertEqual(report.findings[0].status, "supported")

    def test_negation_of_same_predicate_is_conflicting(self):
        report = self.agent.verify({
            "research_results": [{
                "text": "CNN detects threats [1].", "citation_numbers": [1]
            }],
            "research_citations": [{
                "number": 1, "snippet": "CNN does not detect threats in this test."
            }],
        })
        self.assertEqual(report.findings[0].status, "conflicting")

    def test_multiple_cited_passages_can_supply_a_paraphrased_match(self):
        report = self.agent.verify({
            "research_results": [{
                "text": "CNN extracts spatial features [1] [2].",
                "citation_numbers": [1, 2],
            }],
            "research_citations": [
                {"number": 1, "title": "Unrelated", "snippet": "Solar panels convert sunlight."},
                {"number": 2, "title": "Relevant", "url": "https://example.test/2",
                 "snippet": "Convolutional neural networks capture spatial patterns."},
            ],
        })
        finding = report.findings[0]
        self.assertEqual(finding.status, "supported")
        self.assertEqual(len(finding.supporting_sources), 2)
        self.assertEqual(finding.supporting_sources[1]["url"], "https://example.test/2")

    def test_preserved_web_and_document_records_remain_distinguishable(self):
        report = self.agent.verify({
            "research_results": [{
                "text": "The model detects 92% of threats [1].", "citation_numbers": [1]
            }],
            "research_citations": [{
                "number": 1, "title": "Web", "url": "https://example.test",
                "snippet": "The model detects 92% of threats.",
            }],
            "source_evidence": [
                {"source_id": "web:1", "source_type": "web", "citation_id": 1,
                 "source_reference": "https://example.test", "source_title": "Web",
                 "text": "The model detects 92% of threats."},
                {"source_id": "document:d:chunk", "source_type": "document",
                 "document_id": "d", "filename": "report.pdf", "page_number": 9,
                 "chunk_id": "chunk", "text": "The model detects 81% of threats."},
            ],
        })
        finding = report.findings[0]
        self.assertEqual(finding.status, "conflicting")
        self.assertEqual({item["source_type"] for item in finding.supporting_sources}, {"web", "document"})
        document = next(item for item in finding.supporting_sources if item["source_type"] == "document")
        self.assertEqual(document["filename"], "report.pdf")
        self.assertEqual(document["page_number"], 9)


if __name__ == "__main__":
    unittest.main()
