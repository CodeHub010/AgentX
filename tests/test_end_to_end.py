"""Mocked integration tests for the AgentX end-to-end runner."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from agentx.agents.rag_agent import RAGAnswer, RAGSource
from agentx.agents.research_agent import ResearchAnswer, ResearchCitation, ResearchFinding
from agentx.core.end_to_end import DocumentUnavailableError, EndToEndError, run_end_to_end
from agentx.core.settings import Settings
from agentx.storage import StoredChunk
from agentx.tools.web_search import SearchProviderError


def _research_answer() -> ResearchAnswer:
    return ResearchAnswer(
        question="Explain the CNN-LSTM architecture and compare recent web research.",
        summary="CNN extracts features before LSTM models sequences [1].",
        findings=(ResearchFinding("CNN extracts features before LSTM models sequences [1].", (1,)),),
        citations=(ResearchCitation(1, "Demo source", "https://example.test/cnn", None, "CNN extracts features before LSTM models sequences."),),
        search_provider="demo", llm_provider="ollama", llm_model="llama3.2:1b",
    )


def _rag_answer() -> RAGAnswer:
    return RAGAnswer(
        question="Explain the CNN-LSTM architecture and compare recent web research.",
        answer="CNN extracts spatial features and LSTM models temporal sequences [1].",
        sources=(RAGSource(
            number=1, filename="Final Year Report.pdf", page_number=17,
            document_id="doc-123", chunk_id="doc-123:p17:c1", similarity=0.91,
            distance=0.09, text="CNN extracts spatial features and LSTM models temporal sequences.",
        ),),
        status="success",
    )


class EndToEndRunnerTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings()
        self.store = Mock()
        self.store.get_by_document_id.return_value = [StoredChunk(
            document_id="doc-123", filename="Final Year Report.pdf", page_number=17,
            chunk_id="doc-123:p17:c1", text="CNN extracts spatial features.",
            embedding=(0.1, 0.2),
        )]

    def _common_patches(self, research, rag):
        return (
            patch("agentx.core.end_to_end.ChromaVectorStore", return_value=self.store),
            patch("agentx.core.end_to_end.ResearchAgent", return_value=research),
            patch("agentx.core.end_to_end.RAGAgent", return_value=rag),
            patch("agentx.core.end_to_end.create_embedding_service"),
            patch("agentx.core.end_to_end.create_web_search_provider"),
        )

    def test_document_run_executes_combined_graph_and_exports_markdown_and_pdf(self):
        research, rag = Mock(), Mock()
        research.research.return_value = _research_answer()
        rag.answer.return_value = _rag_answer()
        with tempfile.TemporaryDirectory() as temp_dir:
            patches = self._common_patches(research, rag)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = run_end_to_end(
                    "Explain the CNN-LSTM architecture and compare recent web research.",
                    document_id="doc-123", settings=self.settings,
                    report_output_dir=temp_dir, progress=None,
                )
            self.assertEqual(result.state["task_type"], "combined_research")
            self.assertEqual(result.state["route"], "combined")
            self.assertIn("DEMO fixture results", result.state["final_report"])
            self.assertIn("final_report", result.state)
            self.assertTrue(result.markdown_path.is_file())
            self.assertTrue(result.pdf_path.is_file())
            self.assertIn("https://example.test/cnn", result.markdown_path.read_text(encoding="utf-8"))
            self.assertIn("Final Year Report.pdf, page 17", result.markdown_path.read_text(encoding="utf-8"))
            research.research.assert_called_once()
            rag.answer.assert_called_once_with(
                "Explain the CNN-LSTM architecture and compare recent web research.",
                document_id="doc-123",
            )
            self.store.close.assert_called_once()

    def test_without_document_runs_research_route_and_skips_rag(self):
        research, rag = Mock(), Mock()
        research.research.return_value = _research_answer()
        with tempfile.TemporaryDirectory() as temp_dir:
            patches = self._common_patches(research, rag)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = run_end_to_end(
                    "Research recent CNN-LSTM methods online.",
                    settings=self.settings, report_output_dir=temp_dir, progress=None,
                )
            self.assertEqual(result.state["route"], "research")
            rag.answer.assert_not_called()

    def test_web_only_run_does_not_initialize_embeddings_or_chroma(self):
        research = Mock()
        research.research.return_value = _research_answer()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("agentx.core.end_to_end.ChromaVectorStore") as store_factory,
                patch("agentx.core.end_to_end.create_embedding_service") as embedding_factory,
                patch("agentx.core.end_to_end.ResearchAgent", return_value=research),
                patch("agentx.core.end_to_end.create_web_search_provider"),
            ):
                result = run_end_to_end(
                    "Research recent CNN-LSTM methods online.",
                    settings=self.settings,
                    report_output_dir=temp_dir,
                    progress=None,
                )
        self.assertEqual(result.state["route"], "research")
        embedding_factory.assert_not_called()
        store_factory.assert_not_called()

    def test_document_only_mode_forwards_document_and_skips_research(self):
        research, rag = Mock(), Mock()
        rag.answer.return_value = _rag_answer()
        with tempfile.TemporaryDirectory() as temp_dir:
            patches = self._common_patches(research, rag)
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                result = run_end_to_end(
                    "What does the report say about CNN-LSTM?",
                    document_id="doc-123",
                    workflow_mode="document_only",
                    settings=self.settings,
                    report_output_dir=temp_dir,
                    progress=None,
                )
            self.assertEqual(result.state["route"], "knowledge")
            self.assertEqual(result.state["rag_answer"], _rag_answer().answer)
            research.research.assert_not_called()
            rag.answer.assert_called_once_with(
                "What does the report say about CNN-LSTM?", document_id="doc-123"
            )

    def test_missing_document_id_fails_safely_before_workflow(self):
        self.store.get_by_document_id.return_value = []
        with patch("agentx.core.end_to_end.ChromaVectorStore", return_value=self.store):
            with self.assertRaisesRegex(DocumentUnavailableError, "no stored chunks"):
                run_end_to_end(
                    "Research recent CNN-LSTM methods online.",
                    document_id="missing-doc", settings=self.settings, progress=None,
                )
        self.store.close.assert_called_once()

    def test_tavily_failure_falls_back_to_explicitly_labeled_demo_results(self):
        live_research, demo_research, rag = Mock(), Mock(), Mock()
        live_research.research.side_effect = SearchProviderError("request failed")
        demo_research.research.return_value = _research_answer()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("agentx.core.end_to_end.ChromaVectorStore", return_value=self.store),
                patch("agentx.core.end_to_end.ResearchAgent", side_effect=[live_research, demo_research]) as agent_factory,
                patch("agentx.core.end_to_end.RAGAgent", return_value=rag),
                patch("agentx.core.end_to_end.create_embedding_service"),
                patch("agentx.core.end_to_end.create_web_search_provider"),
            ):
                messages = []
                settings = Settings(web_search_provider="tavily", tavily_api_key="secret-token")
                result = run_end_to_end(
                    "Research recent CNN-LSTM methods online.", settings=settings,
                    report_output_dir=temp_dir, progress=messages.append,
                )
            self.assertEqual(agent_factory.call_count, 2)
            self.assertEqual(result.search_mode, "demo_fallback")
            self.assertTrue(any("not live research" in message for message in messages))
            self.assertNotIn("secret-token", "\n".join(messages))
            self.assertTrue(any("labeled demo" in notice for notice in result.notices))

    def test_default_runner_provider_remains_ollama(self):
        research, rag = Mock(), Mock()
        research.research.return_value = _research_answer()
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("agentx.core.end_to_end.ChromaVectorStore", return_value=self.store),
                patch("agentx.core.end_to_end.ResearchAgent", return_value=research) as research_factory,
                patch("agentx.core.end_to_end.RAGAgent", return_value=rag) as rag_factory,
                patch("agentx.core.end_to_end.create_embedding_service"),
                patch("agentx.core.end_to_end.create_web_search_provider"),
            ):
                run_end_to_end(
                    "Research recent CNN-LSTM methods online.",
                    settings=Settings(),
                    report_output_dir=temp_dir, progress=None,
                )
            self.assertEqual(research_factory.call_args.kwargs["settings"].llm_provider, "ollama")
            rag_factory.assert_not_called()

    def test_openai_provider_is_forwarded_to_both_chat_agents(self):
        research, rag = Mock(), Mock()
        research.research.return_value = _research_answer()
        rag.answer.return_value = _rag_answer()
        settings = Settings(llm_provider="openai", openai_api_key="mock-credential")
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("agentx.core.end_to_end.ChromaVectorStore", return_value=self.store),
                patch("agentx.core.end_to_end.ResearchAgent", return_value=research) as research_factory,
                patch("agentx.core.end_to_end.RAGAgent", return_value=rag) as rag_factory,
                patch("agentx.core.end_to_end.create_embedding_service"),
                patch("agentx.core.end_to_end.create_web_search_provider"),
            ):
                run_end_to_end(
                    "Compare the uploaded report with web research.",
                    document_id="doc-123",
                    settings=settings,
                    report_output_dir=temp_dir,
                    progress=None,
                )
            self.assertEqual(research_factory.call_args.kwargs["settings"].llm_provider, "openai")
            self.assertEqual(rag_factory.call_args.kwargs["settings"].llm_provider, "openai")

    def test_openai_provider_without_api_key_fails_before_opening_storage(self):
        with patch("agentx.core.end_to_end.ChromaVectorStore") as store_factory:
            with self.assertRaisesRegex(EndToEndError, "OPENAI_API_KEY is not configured"):
                run_end_to_end(
                    "Research recent CNN-LSTM methods online.",
                    settings=Settings(llm_provider="openai"),
                    progress=None,
                )
        store_factory.assert_not_called()

    def test_invalid_provider_fails_with_configuration_error(self):
        with patch("agentx.core.end_to_end.ChromaVectorStore") as store_factory:
            with self.assertRaisesRegex(EndToEndError, "configuration is invalid"):
                run_end_to_end(
                    "Research recent CNN-LSTM methods online.",
                    settings=Settings(llm_provider="unsupported"),
                    progress=None,
                )
        store_factory.assert_not_called()

    def test_empty_question_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            run_end_to_end("  ", settings=self.settings, progress=None)


if __name__ == "__main__":
    unittest.main()

