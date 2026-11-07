"""Tests for deterministic supervisor task classification."""

import unittest

from agentx.core.supervisor import classify_task


class SupervisorClassificationTests(unittest.TestCase):
    def test_classifies_web_research(self):
        self.assertEqual(
            classify_task("Research recent approaches to AI safety online."),
            "web_research",
        )

    def test_classifies_document_question_answering(self):
        self.assertEqual(
            classify_task("What does the uploaded PDF say about the methodology?"),
            "document_qa",
        )

    def test_classifies_combined_document_and_web_research(self):
        self.assertEqual(
            classify_task("Compare this report with current web research and sources."),
            "combined_research",
        )

    def test_ambiguous_request_is_unknown(self):
        self.assertEqual(classify_task("Can you help me with this?"), "unknown")

    def test_empty_or_non_string_query_is_unknown(self):
        self.assertEqual(classify_task("  \n "), "unknown")
        self.assertEqual(classify_task(None), "unknown")


if __name__ == "__main__":
    unittest.main()
