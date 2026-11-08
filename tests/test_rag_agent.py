"""Mocked tests for evidence-grounded RAG answer generation."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agentx.agents import RAGAgent
from agentx.retrieval import RetrievedChunk


class RAGAgentTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.model = Mock()
        self.model.invoke.return_value = SimpleNamespace(
            content="The model combines temporal and spatial features [1]."
        )
        self.passage = RetrievedChunk(
            text="The CNN extracts spatial features and the LSTM models their sequence.",
            document_id="doc-123",
            filename="report.pdf",
            page_number=12,
            chunk_id="doc-123:p12:c0003",
            distance=0.18,
            similarity=0.82,
        )
        self.retriever.retrieve.return_value = [self.passage]
        self.agent = RAGAgent(retriever=self.retriever, model=self.model)

    def test_successful_grounded_answer_generation(self):
        result = self.agent.answer("How does the model work?")
        self.assertEqual(result.status, "success")
        self.assertEqual(
            result.answer, "The model combines temporal and spatial features [1]."
        )
        self.assertEqual(result.question, "How does the model work?")

    def test_retriever_is_invoked_with_question_and_default_top_k(self):
        self.agent.answer("What did the report say?")
        self.retriever.retrieve.assert_called_once_with(
            "What did the report say?", top_k=5, document_id=None
        )

    def test_gateway_receives_prompt_with_question_evidence_and_citation_format(self):
        self.agent.answer("How does the model work?")
        self.model.invoke.assert_called_once()
        prompt = self.model.invoke.call_args.args[0]
        self.assertIn("How does the model work?", prompt)
        self.assertIn("[1] File: report.pdf; page: 12", prompt)
        self.assertIn("document_id: doc-123", prompt)
        self.assertIn("chunk_id: doc-123:p12:c0003", prompt)
        self.assertIn(self.passage.text, prompt)
        self.assertIn("INSUFFICIENT_EVIDENCE:", prompt)
        self.assertIn("[1] (page 12)", prompt)

    def test_architecture_question_prompt_requests_available_layer_details(self):
        question = "What is the CNN-LSTM model used for in this document?"
        passages = [
            RetrievedChunk(
                text="The model is used to classify activity sequences. The CNN uses "
                     "Conv2D layers with 32 and 64 filters and 3x3 kernels, each "
                     "followed by ReLU and max pooling.",
                document_id="doc-123", filename="report.pdf", page_number=32,
                chunk_id="doc-123:p32:c1", distance=0.1, similarity=0.9,
            ),
            RetrievedChunk(
                text="The pooled CNN output is fed into an LSTM with 128 units. "
                     "A dropout layer with rate 0.2 precedes a dense softmax classifier.",
                document_id="doc-123", filename="report.pdf", page_number=33,
                chunk_id="doc-123:p33:c1", distance=0.12, similarity=0.88,
            ),
        ]
        self.retriever.retrieve.return_value = passages
        self.model.invoke.return_value = (
            "The model is used to classify activity sequences. Its CNN uses 32 and 64 "
            "filters with 3x3 kernels, ReLU, and max pooling [1] (page 32). "
            "The pooled output feeds a 128-unit LSTM, followed by dropout at 0.2 "
            "and a dense softmax classifier [2] (page 33)."
        )

        result = self.agent.answer(question)
        prompt = self.model.invoke.call_args.args[0]

        self.assertEqual(result.status, "success")
        self.assertIn(question, prompt)
        self.assertIn("include the stated component order, data flow", prompt)
        self.assertIn("layer configurations", prompt)
        self.assertIn("never infer or invent a missing configuration", prompt)
        self.assertIn("Conv2D layers with 32 and 64 filters", prompt)
        self.assertIn("LSTM with 128 units", prompt)
        self.assertIn("dropout layer with rate 0.2", prompt)
        self.assertIn("[1] (page 32)", result.answer)
        self.assertIn("[2] (page 33)", result.answer)
        self.assertEqual([source.page_number for source in result.sources], [32, 33])

    def test_ordinary_document_question_requests_a_direct_cited_answer(self):
        self.retriever.retrieve.return_value = [
            RetrievedChunk(
                text="The evaluation used 4,800 labeled samples.",
                document_id="doc-123", filename="report.pdf", page_number=7,
                chunk_id="doc-123:p7:c1", distance=0.1, similarity=0.9,
            )
        ]
        self.model.invoke.return_value = "The evaluation used 4,800 labeled samples [1] (page 7)."

        result = self.agent.answer("How many samples were used?")
        prompt = self.model.invoke.call_args.args[0]

        self.assertEqual(result.status, "success")
        self.assertIn("Answer the user's question directly", prompt)
        self.assertIn("only the evidence passages below as factual sources", prompt)
        self.assertIn("[1] (page 7)", result.answer)
        self.assertNotIn("INSUFFICIENT_EVIDENCE:", result.answer)

    def test_document_id_filter_and_custom_top_k_are_forwarded(self):
        self.agent.answer("question", top_k=3, document_id="doc-123")
        self.retriever.retrieve.assert_called_once_with(
            "question", top_k=3, document_id="doc-123"
        )

    def test_source_metadata_and_similarity_are_preserved(self):
        result = self.agent.answer("question")
        self.assertEqual(len(result.sources), 1)
        source = result.sources[0]
        self.assertEqual(source.number, 1)
        self.assertEqual(source.filename, "report.pdf")
        self.assertEqual(source.page_number, 12)
        self.assertEqual(source.document_id, "doc-123")
        self.assertEqual(source.chunk_id, "doc-123:p12:c0003")
        self.assertEqual(source.text, self.passage.text)
        self.assertAlmostEqual(source.similarity, 0.82)
        self.assertAlmostEqual(source.distance, 0.18)

    def test_empty_question_is_rejected_before_retrieval(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.agent.answer(" \n ")
        self.retriever.retrieve.assert_not_called()
        self.model.invoke.assert_not_called()

    def test_no_results_returns_insufficient_evidence_without_llm_call(self):
        self.retriever.retrieve.return_value = []
        result = self.agent.answer("question")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertIn("could not find relevant passages", result.answer)
        self.assertEqual(result.sources, ())
        self.model.invoke.assert_not_called()

    def test_llm_failure_returns_structured_error_and_sources(self):
        self.model.invoke.side_effect = TimeoutError("model timed out")
        result = self.agent.answer("question")
        self.assertEqual(result.status, "error")
        self.assertIn("language model failed", result.answer)
        self.assertIn("Check the configured provider credentials", result.error_message)
        self.assertEqual(result.sources[0].document_id, "doc-123")

    def test_model_can_report_insufficient_evidence(self):
        self.model.invoke.return_value = SimpleNamespace(
            content="INSUFFICIENT_EVIDENCE: The passages do not address the requested date."
        )
        result = self.agent.answer("When did the event happen?")
        self.assertEqual(result.status, "insufficient_evidence")
        self.assertEqual(
            result.answer, "The passages do not address the requested date."
        )

    def test_valid_citations_remain_and_unknown_citations_are_removed(self):
        self.model.invoke.return_value = "Supported statement [1]. Unsupported reference [9]."
        result = self.agent.answer("question")
        self.assertIn("[1]", result.answer)
        self.assertNotIn("[9]", result.answer)
        self.assertEqual(result.sources[0].number, 1)

    def test_retrieval_failure_returns_error_without_calling_llm(self):
        self.retriever.retrieve.side_effect = OSError("Chroma unavailable")
        result = self.agent.answer("question")
        self.assertEqual(result.status, "error")
        self.assertIn("Chroma unavailable", result.error_message)
        self.model.invoke.assert_not_called()

    def test_invalid_top_k_is_rejected(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(top_k=invalid):
                with self.assertRaises(ValueError):
                    RAGAgent(retriever=self.retriever, model=self.model, top_k=invalid)
        with self.assertRaises(ValueError):
            self.agent.answer("question", top_k=0)


if __name__ == "__main__":
    unittest.main()
