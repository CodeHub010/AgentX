"""Tests connecting the existing RAG Agent to the LangGraph workflow."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from agentx.agents import RAGAgent
from agentx.core.workflow import build_workflow
from agentx.retrieval import RetrievedChunk


class WorkflowRAGTests(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock()
        self.model = Mock()
        self.model.invoke.return_value = SimpleNamespace(
            content="The report describes a CNN followed by an LSTM [1]."
        )
        self.retriever.retrieve.return_value = [
            RetrievedChunk(
                text="CNN features are processed by an LSTM.",
                document_id="doc-123",
                filename="report.pdf",
                page_number=12,
                chunk_id="doc-123:p12:c0003",
                distance=0.18,
                similarity=0.82,
            )
        ]
        self.agent = RAGAgent(retriever=self.retriever, model=self.model)

    def test_rag_agent_invocation_stores_answer_metadata_and_citations(self):
        query = "What does the uploaded PDF say about CNN-LSTM?"
        result = build_workflow(rag_agent=self.agent).invoke(
            {"user_query": query, "document_id": "doc-123"}
        )

        self.assertEqual(result["route"], "knowledge")
        self.assertEqual(result["status"], "knowledge_complete")
        self.assertEqual(
            result["rag_answer"],
            "The report describes a CNN followed by an LSTM [1].",
        )
        self.retriever.retrieve.assert_called_once_with(
            query, top_k=5, document_id="doc-123"
        )
        source = result["retrieved_context"][0]
        self.assertEqual(source["filename"], "report.pdf")
        self.assertEqual(source["page_number"], 12)
        self.assertEqual(source["document_id"], "doc-123")
        self.assertEqual(source["chunk_id"], "doc-123:p12:c0003")
        self.assertAlmostEqual(source["similarity"], 0.82)
        evidence = result["source_evidence"][0]
        self.assertEqual(evidence["source_type"], "document")
        self.assertEqual(evidence["source_reference"], "doc-123:p12:c0003")
        self.assertEqual(evidence["text"], self.retriever.retrieve.return_value[0].text)
        self.assertEqual(evidence["filename"], "report.pdf")
        self.assertEqual(evidence["page_number"], 12)
        self.assertEqual(
            result["rag_citations"],
            [{"number": 1, "text": "[1] report.pdf, page 12"}],
        )

    def test_optional_document_id_is_forwarded_as_none_when_not_provided(self):
        query = "What does this PDF say about its methods?"
        build_workflow(rag_agent=self.agent).invoke({"user_query": query})
        self.retriever.retrieve.assert_called_once_with(
            query, top_k=5, document_id=None
        )

    def test_insufficient_evidence_status_and_answer_are_preserved(self):
        self.retriever.retrieve.return_value = []
        result = build_workflow(rag_agent=self.agent).invoke(
            {"user_query": "What does the report say about this?"}
        )

        self.assertEqual(result["status"], "knowledge_insufficient_evidence")
        self.assertIn("could not find relevant passages", result["rag_answer"])
        self.assertEqual(result["retrieved_context"], [])
        self.assertEqual(result["rag_citations"], [])
        self.model.invoke.assert_not_called()

    def test_retrieval_or_model_error_is_recorded_safely(self):
        self.retriever.retrieve.side_effect = OSError("storage unavailable")
        result = build_workflow(rag_agent=self.agent).invoke(
            {"user_query": "What does the PDF say about this?"}
        )

        self.assertEqual(result["status"], "knowledge_error")
        self.assertEqual(result["rag_answer"], "I could not retrieve evidence for this question.")
        self.assertIn("Retrieval failed", result["errors"][-1])
        self.assertIn("storage unavailable", result["errors"][-1])

    def test_llm_error_is_recorded_and_source_metadata_is_retained(self):
        self.model.invoke.side_effect = TimeoutError("model timed out")
        result = build_workflow(rag_agent=self.agent).invoke(
            {"user_query": "What does the PDF say about this?"}
        )

        self.assertEqual(result["status"], "knowledge_error")
        self.assertIn("language model failed", result["rag_answer"])
        self.assertIn("LLM request failed", result["errors"][-1])
        self.assertEqual(result["retrieved_context"][0]["document_id"], "doc-123")
        self.assertEqual(result["rag_citations"][0]["text"], "[1] report.pdf, page 12")

    def test_unexpected_rag_exception_is_caught_by_workflow_node(self):
        agent = Mock()
        agent.answer.side_effect = RuntimeError("unexpected failure")
        result = build_workflow(rag_agent=agent).invoke(
            {"user_query": "What does the PDF say?"}
        )

        self.assertEqual(result["status"], "knowledge_error")
        self.assertIn("unexpected failure", result["errors"][-1])


if __name__ == "__main__":
    unittest.main()
