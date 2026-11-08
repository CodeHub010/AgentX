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

    def test_document_headings_and_nonfactual_introductions_are_not_claims(self):
        report = self.agent.verify({
            "rag_answer": "## Uploaded Document Findings\nHere is a summary.\n"
                          "CNN identifies spatial patterns [1].",
            "retrieved_context": [{
                "number": 1, "filename": "report.pdf", "page_number": 4,
                "document_id": "doc-4", "chunk_id": "doc-4:p4:c1",
                "text": "Convolutional neural networks recognize spatial patterns.",
            }],
        })
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].status, "supported")
        source = report.findings[0].supporting_sources[0]
        self.assertEqual(source["filename"], "report.pdf")
        self.assertEqual(source["page_number"], 4)
        self.assertEqual(source["document_id"], "doc-4")
        self.assertEqual(source["chunk_id"], "doc-4:p4:c1")

    def test_partial_keyword_overlap_is_not_enough_for_support(self):
        report = self.agent.verify({
            "research_results": [{
                "text": "CNN prevents all insider attacks in every network [1].",
                "citation_numbers": [1],
            }],
            "research_citations": [{
                "number": 1,
                "snippet": "CNN models are studied for detecting insider threats in network traffic.",
            }],
        })
        self.assertNotEqual(report.findings[0].status, "supported")

    def test_insufficient_rag_response_does_not_turn_heading_into_claim(self):
        report = self.agent.verify({
            "rag_answer": "## Evidence Verification\nHere is a summary.",
            "rag_status": "insufficient_evidence",
            "retrieved_context": [],
        })
        self.assertEqual(report.findings[0].status, "insufficient_evidence")
        self.assertIn("no claim text", report.findings[0].explanation)

    def test_comma_separated_citations_are_parsed_and_sources_preserved(self):
        for citation in ("[1, 2]", "[1,2]", "[1, 2, 3]"):
            with self.subTest(citation=citation):
                numbers = self.agent._reference_numbers(None, f"Claim {citation}.")
                expected = tuple(int(number.strip()) for number in citation[1:-1].split(","))
                self.assertEqual(numbers, expected)

        report = self.agent.verify({
            "rag_answer": "CNN detects threats [1, 2].",
            "retrieved_context": [
                {"number": 1, "filename": "a.pdf", "page_number": 3,
                 "document_id": "doc-a", "chunk_id": "a:p3:c1",
                 "text": "CNN detects threats in network traffic."},
                {"number": 2, "filename": "b.pdf", "page_number": 5,
                 "document_id": "doc-b", "chunk_id": "b:p5:c2",
                 "text": "CNN detects threats in network traffic."},
            ],
        })
        finding = report.findings[0]
        self.assertEqual(finding.status, "supported")
        self.assertEqual(
            [(source["filename"], source["page_number"], source["document_id"], source["chunk_id"])
             for source in finding.supporting_sources],
            [("a.pdf", 3, "doc-a", "a:p3:c1"), ("b.pdf", 5, "doc-b", "b:p5:c2")],
        )

    def test_citation_on_following_line_stays_with_its_claim(self):
        claims = self.agent._split_claims(
            "- CNN detects threats in networks.\n  [1, 2]\n"
            "- LSTM processes temporal sequences [3]."
        )
        self.assertEqual(len(claims), 2)
        self.assertEqual(self.agent._reference_numbers(None, claims[0]), (1, 2))
        self.assertEqual(self.agent._reference_numbers(None, claims[1]), (3,))

    def test_bullets_and_markdown_emphasis_preserve_inline_citations(self):
        claims = self.agent._split_claims(
            "- Claim [1]\n• Claim [2]\n- **Claim [3]**"
        )
        self.assertEqual(len(claims), 3)
        self.assertEqual(
            [self.agent._reference_numbers(None, claim) for claim in claims],
            [(1,), (2,), (3,)],
        )

    def test_multiple_claims_keep_their_own_citation_references(self):
        claims = self.agent._split_claims(
            "CNN detects threats [1]. LSTM processes sequences [2, 3]."
        )
        self.assertEqual(len(claims), 2)
        self.assertEqual(
            [self.agent._reference_numbers(None, claim) for claim in claims],
            [(1,), (2, 3)],
        )

    def test_decimal_and_common_abbreviation_periods_do_not_split_claims(self):
        claims = self.agent._split_claims(
            "The model reaches 3.14 accuracy, e.g. on CNN data, according to Dr. Lee [1]."
        )
        self.assertEqual(len(claims), 1)
        self.assertIn("3.14", claims[0])
        self.assertIn("e.g.", claims[0])
        self.assertIn("Dr. Lee", claims[0])
        self.assertEqual(self.agent._reference_numbers(None, claims[0]), (1,))

    def test_cnn_lstm_acronyms_and_expansions_match_explicit_transfer_phrases(self):
        for claim_phrase, evidence_phrase in (
            ("is passed to", "is fed into"),
            ("is fed into", "is transferred to"),
            ("is transferred to", "is passed to"),
        ):
            with self.subTest(claim_phrase=claim_phrase, evidence_phrase=evidence_phrase):
                report = self.agent.verify({
                    "rag_answer": (
                        "Convolutional Neural Network output "
                        f"{claim_phrase} Long Short-Term Memory [1]."
                    ),
                    "retrieved_context": [{
                        "number": 1, "filename": "architecture.pdf", "page_number": 32,
                        "document_id": "doc-32", "chunk_id": "doc-32:p32:c1",
                        "text": f"CNN output {evidence_phrase} LSTM.",
                    }],
                })
                finding = report.findings[0]
                self.assertEqual(finding.status, "supported")
                self.assertEqual(finding.supporting_sources[0]["document_id"], "doc-32")
                self.assertEqual(finding.supporting_sources[0]["page_number"], 32)

    def test_adjacent_sentences_can_supply_a_bounded_cnn_lstm_explanation(self):
        report = self.agent.verify({
            "rag_answer": "CNN output is passed to LSTM [1].",
            "retrieved_context": [{
                "number": 1, "filename": "architecture.pdf", "page_number": 32,
                "document_id": "doc-32", "chunk_id": "doc-32:p32:c2",
                "text": "The convolutional neural network produces an output. "
                        "That output is transferred to the long short-term memory layer.",
            }],
        })
        finding = report.findings[0]
        self.assertEqual(finding.status, "supported")
        self.assertEqual(finding.supporting_sources[0]["filename"], "architecture.pdf")
        self.assertEqual(finding.supporting_sources[0]["chunk_id"], "doc-32:p32:c2")

    def test_reversed_cnn_lstm_transfer_is_not_supported(self):
        report = self.agent.verify({
            "rag_answer": "CNN output is passed to LSTM [1].",
            "retrieved_context": [{
                "number": 1,
                "text": "LSTM output is passed to CNN.",
            }],
        })
        self.assertNotEqual(report.findings[0].status, "supported")

    def test_unrelated_cnn_and_lstm_mentions_do_not_support_transfer_claim(self):
        report = self.agent.verify({
            "rag_answer": "CNN output is transferred to LSTM [1].",
            "retrieved_context": [{
                "number": 1,
                "text": "CNN classifies images. LSTM independently predicts weather.",
            }],
        })
        self.assertNotEqual(report.findings[0].status, "supported")

    def test_citation_and_page_numbers_do_not_create_numeric_conflicts(self):
        cases = (
            (
                "Output Activation: Softmax [3] (page 34)",
                "The output activation is Softmax.", 3,
            ),
            (
                "Loss Function: Categorical Cross-Entropy [3] (page 34)",
                "Categorical cross-entropy is used as the loss function.", 3,
            ),
            (
                "Framework: Python using TensorFlow and Keras libraries [4] (page 36)",
                "The framework uses Python with TensorFlow and Keras libraries.", 4,
            ),
        )
        for claim, source_text, citation in cases:
            with self.subTest(claim=claim):
                report = self.agent.verify({
                    "rag_answer": claim,
                    "retrieved_context": [{"number": citation, "page_number": 34,
                                           "text": source_text}],
                })
                self.assertNotEqual(report.findings[0].status, "conflicting")

    def test_multiple_page_citations_do_not_conflict_with_each_other(self):
        report = self.agent.verify({
            "rag_answer": (
                "The LSTM component captures behavioral changes, temporal "
                "relationships, and sequential dependencies [1] (page 33), "
                "[2] (page 32)"
            ),
            "retrieved_context": [
                {"number": 1, "page_number": 33,
                 "text": "The LSTM captures behavioral changes and temporal relationships."},
                {"number": 2, "page_number": 32,
                 "text": "The LSTM models sequential dependencies."},
            ],
        })
        self.assertNotEqual(report.findings[0].status, "conflicting")
        self.assertEqual(
            {source["number"] for source in report.findings[0].supporting_sources},
            {1, 2},
        )

    def test_numeric_conflict_matches_the_same_property_not_nearby_numbers(self):
        report = self.agent.verify({
            "rag_answer": "The LSTM has 32 units [1].",
            "retrieved_context": [{
                "number": 1,
                "text": "The LSTM uses 64 units, while the CNN has 32 layers.",
            }],
        })
        finding = report.findings[0]
        self.assertEqual(finding.status, "conflicting")
        self.assertIn("32.0 versus 64.0", finding.contradiction)

    def test_document_and_chunk_identifiers_are_not_numeric_facts(self):
        facts = self.agent._numeric_facts(
            "Claim [1, 2] (page 33); document_id: doc-2024; "
            "chunk_id: doc-2024:p33:c0002"
        )
        self.assertEqual(facts, [])

    def test_configuration_numbers_compare_only_by_explicit_property(self):
        cases = (
            ("| Input shape | (1, 78) |", "| Input shape | (1, 78) |\n| Conv1D filters | 64 |"),
            ("| Conv1D | filters=64 | kernel_size=3 |", "| Conv1D | filters=64 | kernel_size=3 |\n| Pooling | pool_size=2 |"),
            ("| Max Pooling | pool_size=2 |", "| Max Pooling | pool_size=2 |\n| Conv1D | filters=64 |"),
            ("| LSTM | units | 64 |", "| LSTM | units | 64 |\n| Dense | neurons | 32 |"),
            ("| Dropout | rate=0.5 |", "| Dropout | rate=0.5 |\n| Layer | 3 | filters=64 |"),
        )
        for claim, evidence in cases:
            with self.subTest(claim=claim):
                self.assertIsNone(self.agent._best_contradiction(claim, [evidence]))

    def test_markdown_table_same_property_mismatch_is_a_conflict(self):
        cases = (
            ("| Input shape | (1, 78) |", "| Input shape | (1, 64) |"),
            ("| Conv1D | filters=64 | kernel_size=3 |", "| Conv1D | filters=64 | kernel_size=5 |"),
            ("| Max Pooling | pool_size=2 |", "| Max Pooling | pool_size=3 |"),
            ("| LSTM | units | 32 |", "| LSTM | units | 64 |"),
            ("| Dropout | rate=0.5 |", "| Dropout | rate=0.25 |"),
            ("| Dense | neurons | 32 |", "| Dense | neurons | 64 |"),
        )
        for claim, evidence in cases:
            with self.subTest(claim=claim):
                contradiction = self.agent._best_contradiction(claim, [evidence])
                self.assertIsNotNone(contradiction)
                self.assertIn("same property", contradiction)

    def test_lstm_units_conflict_does_not_use_other_table_values(self):
        report = self.agent.verify({
            "rag_answer": "| LSTM | units | 32 | [1]",
            "retrieved_context": [{
                "number": 1,
                "text": "| Layer | Type | Property | Value |\n"
                        "| 1 | CNN | filters | 32 |\n"
                        "| 2 | LSTM | units | 64 |",
            }],
        })
        self.assertEqual(report.findings[0].status, "conflicting")
        self.assertIn("lstm_units", report.findings[0].contradiction)

    def test_markdown_cnn_lstm_report_rows_stay_whole_and_keep_citations(self):
        answer = (
            "## CNN-LSTM Architecture\n---\n"
            "| Layer | Configuration | Citation |\n"
            "|---|---|---|\n"
            "| Input Shape | (1, 78) | [1] |\n"
            "| Conv1D Filters | 64 | [2] |\n"
            "| Kernel Size | 3 |\n[3]\n"
            "| Max Pooling Size | 2 | [4] |\n"
            "| LSTM Layer - Units: 64 | [5] |\n"
            "| Dropout Rate | 0.5 | [6] |\n"
            "| Dense Neurons | 32 | [7] |\n"
            "| Output Classes | 15 | [8] |\n"
            "| Output Activation | Softmax | [9] |"
        )
        claims = self.agent._split_claims(answer)
        self.assertEqual(len(claims), 9)
        self.assertTrue(all("|" in claim for claim in claims))
        self.assertEqual(
            [self.agent._reference_numbers(None, claim) for claim in claims],
            [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)],
        )
        self.assertTrue(all("---" not in claim for claim in claims))

    def test_markdown_rows_verify_against_matching_configuration_property(self):
        rows = (
            ("| Input Shape | (1, 78) |", "Input shape is (1, 78)."),
            ("| Conv1D Filters | 64 |", "Conv1D uses 64 filters."),
            ("| Kernel Size | 3 |", "The kernel size is 3."),
            ("| Max Pooling Size | 2 |", "Max pooling uses a pool size of 2."),
            ("| LSTM Layer - Units: 64 |", "The LSTM layer has 64 units."),
            ("| Dropout Rate | 0.5 |", "The dropout rate is 0.5."),
            ("| Dense Neurons | 32 |", "The dense layer has 32 neurons."),
            ("| Output Classes | 15 |", "The output contains 15 classes."),
            ("| Output Activation | Softmax |", "The output activation is Softmax."),
        )
        table = "\n".join(f"{claim} [{index}] |" for index, (claim, _) in enumerate(rows, 1))
        sources = [
            {"number": index, "filename": "cnn_lstm.pdf", "page_number": 20 + index,
             "text": evidence}
            for index, (_, evidence) in enumerate(rows, 1)
        ]
        report = self.agent.verify({"rag_answer": table, "retrieved_context": sources})
        self.assertEqual(len(report.findings), 9)
        self.assertTrue(all(finding.status == "supported" for finding in report.findings))
        self.assertEqual(
            [finding.supporting_sources[0]["page_number"] for finding in report.findings],
            list(range(21, 30)),
        )

    def test_markdown_lstm_units_mismatch_remains_conflicting(self):
        report = self.agent.verify({
            "rag_answer": "| LSTM Layer - Units: 32 | [1] |",
            "retrieved_context": [{
                "number": 1, "filename": "cnn_lstm.pdf", "page_number": 25,
                "text": "The LSTM layer has 64 units.",
            }],
        })
        finding = report.findings[0]
        self.assertEqual(finding.status, "conflicting")
        self.assertIn("lstm_units", finding.contradiction)

    def test_concise_configuration_claims_match_explicit_property_and_value(self):
        cases = (
            ("Filters: 64", "64 filters", 33),
            ("Kernel Size: 3", "kernel size = 3", 33),
            ("LSTM Units: 64", "LSTM Layer 64 units", 33),
            ("Input Shape: (1, 78)", "Input shape: (1,78)", 33),
            ("Batch Size: 32", "Batch Size 32", 34),
            ("Epochs: 20", "Epochs 20", 34),
            (
                "Train/Test Split: 80% training / 20% testing",
                "Training data: 80%, Testing data: 20%",
                36,
            ),
        )
        for claim, passage, page in cases:
            with self.subTest(claim=claim):
                report = self.agent.verify({
                    "rag_answer": f"{claim} [1]",
                    "retrieved_context": [{
                        "number": 1, "filename": "architecture.pdf", "page_number": page,
                        "document_id": "doc-cnn-lstm", "chunk_id": f"p{page}:c1",
                        "text": passage,
                    }],
                })
                finding = report.findings[0]
                self.assertEqual(finding.status, "supported")
                self.assertEqual(finding.supporting_sources[0]["page_number"], page)
                self.assertEqual(finding.supporting_sources[0]["document_id"], "doc-cnn-lstm")

    def test_concise_numeric_claims_reject_wrong_property_and_detect_wrong_value(self):
        wrong_property = self.agent.verify({
            "rag_answer": "Filters: 64 [1]",
            "retrieved_context": [{"number": 1, "text": "LSTM Layer 64 units."}],
        })
        self.assertNotEqual(wrong_property.findings[0].status, "supported")

        wrong_value = self.agent.verify({
            "rag_answer": "Conv1D Filters: 32 [1]",
            "retrieved_context": [{"number": 1, "text": "Conv1D uses 64 filters."}],
        })
        self.assertEqual(wrong_value.findings[0].status, "conflicting")

        unrelated_value = self.agent.verify({
            "rag_answer": "Batch Size: 32 [1]",
            "retrieved_context": [{"number": 1, "text": "Epochs 32."}],
        })
        self.assertNotEqual(unrelated_value.findings[0].status, "supported")

    def test_short_lstm_units_claim_with_wrong_value_still_conflicts(self):
        report = self.agent.verify({
            "rag_answer": "LSTM Units: 32 [1]",
            "retrieved_context": [{"number": 1, "text": "LSTM Layer 64 units."}],
        })
        self.assertEqual(report.findings[0].status, "conflicting")

    def test_configuration_diagnostics_cover_all_reported_claim_shapes(self):
        cases = (
            ("Filters: 64", "64 filters", "filters"),
            ("Kernel Size: 3", "kernel size = 3", "kernel_size"),
            ("LSTM Units: 64", "LSTM Layer 64 units", "lstm_units"),
            ("Dropout Rate: 0.5", "Dropout rate = 0.5", "dropout_rate"),
            ("Neurons: 32", "32 neurons", "neurons"),
            ("Batch Size: 32", "Batch Size 32", "batch_size"),
            ("Epochs: 20", "Epochs 20", "epochs"),
            ("Input Shape: (1, 78)", "Input shape: (1,78)", "input_shape"),
        )
        for claim, passage, property_key in cases:
            with self.subTest(claim=claim):
                claim_facts = self.agent._configuration_facts(claim)
                evidence_facts = self.agent._configuration_facts(passage)
                self.assertTrue(claim_facts, f"No explicit claim property parsed: {claim}")
                self.assertIn(
                    property_key,
                    claim_facts if property_key in claim_facts else evidence_facts,
                )
                report = self.agent.verify({
                    "rag_answer": f"{claim} [1]",
                    "retrieved_context": [{
                        "number": 1, "filename": "model.pdf", "page_number": 33,
                        "document_id": "doc-33", "chunk_id": "doc-33:p33:c1",
                        "text": passage,
                    }],
                })
                self.assertEqual(report.findings[0].status, "supported")
                source = report.findings[0].supporting_sources[0]
                self.assertEqual((source["filename"], source["page_number"], source["document_id"]),
                                 ("model.pdf", 33, "doc-33"))

    def test_wrapped_pdf_table_cells_keep_property_value_association(self):
        # PyPDF extraction can split table labels and values across physical
        # lines. _evidence_units joins only neighboring lines; the parser must
        # preserve that explicit property/value association after the join.
        report = self.agent.verify({
            "rag_answer": "Filters: 64 [1]",
            "retrieved_context": [{
                "number": 1, "filename": "architecture.pdf", "page_number": 33,
                "document_id": "doc-33", "chunk_id": "doc-33:p33:c2",
                "text": "Conv1D Layer\nFilters\n64\nKernel Size\n3",
            }],
        })
        finding = report.findings[0]
        self.assertEqual(finding.status, "supported")
        self.assertEqual(finding.supporting_sources[0]["chunk_id"], "doc-33:p33:c2")

    def test_neurons_wrong_value_and_wrong_property_are_not_supported(self):
        mismatch = self.agent.verify({
            "rag_answer": "Dense Neurons: 32 [1]",
            "retrieved_context": [{"number": 1, "text": "Dense layer has 64 neurons."}],
        })
        self.assertEqual(mismatch.findings[0].status, "conflicting")

        wrong_property = self.agent.verify({
            "rag_answer": "Dense Neurons: 32 [1]",
            "retrieved_context": [{"number": 1, "text": "LSTM layer has 32 units."}],
        })
        self.assertNotEqual(wrong_property.findings[0].status, "supported")

    def test_unavailable_intermediate_dimension_note_is_not_a_numeric_conflict(self):
        report = self.agent.verify({
            "rag_answer": "The intermediate feature dimension is unavailable [1].",
            "retrieved_context": [{
                "number": 1,
                "text": "The intermediate feature dimension is not reported. The output layer has 15 neurons.",
            }],
        })
        self.assertNotEqual(report.findings[0].status, "conflicting")

    def test_dense_and_output_neuron_counts_are_layer_specific(self):
        common_evidence = [{
            "number": 1, "filename": "architecture.pdf", "page_number": 33,
            "document_id": "doc-33", "chunk_id": "doc-33:p33:c1",
            "text": "Dense layer has 32 neurons. Output layer has 15 neurons.",
        }]
        dense = self.agent.verify({
            "rag_answer": "Dense layer has 32 neurons [1]", "retrieved_context": common_evidence,
        }).findings[0]
        output = self.agent.verify({
            "rag_answer": "Output layer has 15 neurons [1]", "retrieved_context": common_evidence,
        }).findings[0]
        self.assertEqual(dense.status, "supported")
        self.assertEqual(output.status, "supported")
        self.assertFalse(self.agent._best_contradiction(
            "Dense layer has 32 neurons", ["Output layer has 15 neurons"]
        ))

    def test_output_and_dense_neuron_contradictions_are_layer_scoped(self):
        cases = (
            ("Output layer has 15 neurons [1]", "Output layer has 32 neurons", "output_neurons"),
            ("Dense layer has 32 neurons [1]", "Dense layer has 64 neurons", "dense_neurons"),
            ("LSTM layer has 64 units [1]", "LSTM layer has 32 units", "lstm_units"),
        )
        for claim, evidence, property_name in cases:
            with self.subTest(claim=claim):
                finding = self.agent.verify({
                    "rag_answer": claim,
                    "retrieved_context": [{"number": 1, "text": evidence}],
                }).findings[0]
                self.assertEqual(finding.status, "conflicting")
                self.assertIn(property_name, finding.contradiction)

    def test_wrong_layer_cannot_support_same_neuron_count(self):
        finding = self.agent.verify({
            "rag_answer": "Dense layer has 32 neurons [1]",
            "retrieved_context": [{
                "number": 1, "text": "Output layer has 32 neurons.",
            }],
        }).findings[0]
        self.assertNotEqual(finding.status, "supported")

    def test_multiple_layer_properties_do_not_cross_match_values(self):
        claim = "Dense layer has 32 neurons and Output layer has 15 neurons [1]"
        evidence = "Dense layer has 32 neurons. Output layer has 15 neurons."
        finding = self.agent.verify({
            "rag_answer": claim,
            "retrieved_context": [{"number": 1, "text": evidence}],
        }).findings[0]
        self.assertEqual(finding.status, "supported")
        self.assertFalse(self.agent._best_contradiction(
            claim.removesuffix(" [1]"), [evidence]
        ))

    def test_additional_hyperparameters_use_exact_property_matching(self):
        cases = (
            ("Input Shape: (1, 78)", "Input shape is (1,78)", "input_shape"),
            ("LSTM Units: 64", "LSTM Layer 64 units", "lstm_units"),
            ("Batch Size: 32", "Batch Size 32", "batch_size"),
            ("Epochs: 20", "Epochs 20", "epochs"),
            ("Optimizer: Adam", "Adam optimizer", "optimizer"),
            ("Learning Rate: 0.001", "Learning rate 0.001", "learning_rate"),
        )
        for claim, evidence, property_name in cases:
            with self.subTest(claim=claim):
                self.assertEqual(self.agent._configuration_facts(claim).get(property_name),
                                 self.agent._configuration_facts(evidence).get(property_name))
                finding = self.agent.verify({
                    "rag_answer": f"{claim} [1]",
                    "retrieved_context": [{"number": 1, "text": evidence}],
                }).findings[0]
                self.assertEqual(finding.status, "supported")

        optimizer_mismatch = self.agent.verify({
            "rag_answer": "Optimizer: Adam [1]",
            "retrieved_context": [{"number": 1, "text": "SGD optimizer"}],
        }).findings[0]
        self.assertEqual(optimizer_mismatch.status, "conflicting")

    def test_markdown_structure_is_not_emitted_as_claims(self):
        claims = self.agent._split_claims(
            "## Architecture Summary\n***\n\n"
            "| Layer | Property | Value |\n|---|---|---|\n"
            "| Filters | 64 | [1] |"
        )
        self.assertEqual(claims, ["| Filters | 64 | [1] |"])

    def test_layer_heading_context_is_preserved_for_all_supported_layers(self):
        cases = (
            ("Layer 1: Input Layer", "Input Shape: (1, 78)", "input_shape"),
            ("Layer 2: Conv1D Layer", "Filters: 64", "conv1d_filters"),
            ("Layer 3: Max Pooling Layer", "Pool Size: 2", "max_pool_size"),
            ("Layer 4: LSTM Layer", "Units: 64", "lstm_units"),
            ("Layer 5: Dropout Layer", "Dropout Rate: 0.5", "dropout_rate"),
            ("Layer 6: Dense Layer", "Neurons: 32", "dense_neurons"),
            ("Layer 7: Output Layer", "Neurons: 15", "output_neurons"),
        )
        for heading, row, expected_property in cases:
            with self.subTest(layer=heading):
                claims = self.agent._split_claims(f"{heading}\n\n- {row}\n[1]")
                self.assertEqual(len(claims), 1)
                claim = claims[0]
                self.assertIn(heading.split(": ", 1)[1], claim)
                self.assertIn(row, claim)
                self.assertIn(1, self.agent._reference_numbers(None, claim))
                self.assertIn(expected_property, self.agent._configuration_facts(claim))

    def test_layer_context_resets_at_next_layer_and_section_boundary(self):
        claims = self.agent._split_claims(
            "Layer 6: Dense Layer\nNeurons: 32 [1]\n"
            "Layer 7: Output Layer\nNeurons: 15 [2]\n"
            "## Training Configuration\nBatch Size: 32 [3]"
        )
        self.assertEqual(len(claims), 3)
        self.assertIn("Dense Layer", claims[0])
        self.assertIn("Output Layer", claims[1])
        self.assertNotIn("Output Layer", claims[2])
        self.assertNotIn("Dense Layer", claims[2])

    def test_dense_and_output_heading_claims_are_independently_supported(self):
        evidence = [{
            "number": 1,
            "filename": "architecture.pdf",
            "page_number": 33,
            "document_id": "doc-33",
            "chunk_id": "doc-33:p33:c1",
            "text": "Dense layer has 32 neurons. Output layer has 15 neurons.",
        }]
        report = self.agent.verify({
            "rag_answer": "Layer 6: Dense Layer\nNeurons: 32 [1]\n"
                          "Layer 7: Output Layer\nNeurons: 15 [1]",
            "retrieved_context": evidence,
        })
        self.assertEqual([finding.status for finding in report.findings], ["supported", "supported"])
        self.assertEqual(report.findings[0].supporting_sources[0]["page_number"], 33)
        self.assertEqual(report.findings[1].supporting_sources[0]["document_id"], "doc-33")

    def test_output_heading_keeps_genuine_neuron_contradiction(self):
        report = self.agent.verify({
            "rag_answer": "Layer 7: Output Layer\nNeurons: 32 [1]",
            "retrieved_context": [{"number": 1, "text": "Output layer has 15 neurons."}],
        })
        self.assertEqual(report.findings[0].status, "conflicting")
        self.assertIn("output_neurons", report.findings[0].contradiction)

    def test_layer_purpose_is_insufficient_when_source_only_has_configuration(self):
        cases = (
            ("Max Pooling reduces spatial dimensions [1]", "Max Pooling pool size 2."),
            ("Dropout prevents overfitting [1]", "Dropout rate 0.5."),
        )
        for claim, evidence in cases:
            with self.subTest(claim=claim):
                finding = self.agent.verify({
                    "rag_answer": claim,
                    "retrieved_context": [{"number": 1, "text": evidence}],
                }).findings[0]
                self.assertEqual(finding.status, "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
