"""Grounded question answering over PDF chunks retrieved from ChromaDB."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from agentx.core.settings import Settings
from agentx.llm import ChatModel, create_chat_model
from agentx.retrieval import RetrievedChunk, SemanticRetriever

RAGStatus = Literal["success", "insufficient_evidence", "error"]
_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_INSUFFICIENT_PREFIX = "INSUFFICIENT_EVIDENCE:"


@dataclass(frozen=True, slots=True)
class RAGSource:
    """A retrieved evidence passage with its exact PDF source metadata."""

    number: int
    filename: str
    page_number: int
    document_id: str
    chunk_id: str
    similarity: float | None
    distance: float | None
    text: str | None = None


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Structured response from the AgentX PDF question-answering agent."""

    question: str
    answer: str
    sources: tuple[RAGSource, ...]
    status: RAGStatus
    error_message: str | None = None


class RAGAgent:
    """Retrieve PDF evidence and synthesize an answer through AgentX's gateway.

    Retrieval and language-model dependencies can be injected for tests. When
    no passages are retrieved, the model is not invoked.
    """

    DEFAULT_TOP_K = 5

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        retriever: SemanticRetriever | None = None,
        model: ChatModel | None = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self._validate_top_k(top_k)
        self._settings = settings
        self._retriever = retriever or SemanticRetriever(settings=settings, top_k=top_k)
        self._model = model
        self.top_k = top_k

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> RAGAnswer:
        """Answer a question only from retrieved PDF passages."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("RAG question must not be empty.")
        selected_question = question.strip()
        selected_top_k = self.top_k if top_k is None else top_k
        self._validate_top_k(selected_top_k)

        try:
            passages = self._retriever.retrieve(
                selected_question,
                top_k=selected_top_k,
                document_id=document_id,
            )
        except Exception as exc:
            return RAGAnswer(
                question=selected_question,
                answer="I could not retrieve evidence for this question.",
                sources=(),
                status="error",
                error_message=f"Retrieval failed ({type(exc).__name__}): {exc}",
            )

        if not passages:
            return RAGAnswer(
                question=selected_question,
                answer=(
                    "I could not find relevant passages in the stored PDF documents "
                    "to answer this question."
                ),
                sources=(),
                status="insufficient_evidence",
            )

        sources = tuple(self._source(index, passage) for index, passage in enumerate(passages, 1))
        prompt = self._build_prompt(selected_question, passages, sources)
        try:
            model = self._model or create_chat_model(settings=self._settings)
            answer = self._response_text(model.invoke(prompt))
            if not answer:
                raise ValueError("The model returned an empty response.")
        except Exception as exc:
            return RAGAnswer(
                question=selected_question,
                answer="I could not generate an answer because the language model failed.",
                sources=sources,
                status="error",
                error_message=f"LLM request failed ({type(exc).__name__}): {exc}",
            )

        status = "success"
        if answer.upper().startswith(_INSUFFICIENT_PREFIX):
            answer = answer[len(_INSUFFICIENT_PREFIX):].strip()
            status = "insufficient_evidence"
            if not answer:
                answer = "The retrieved passages do not provide enough evidence to answer."
        answer = self._remove_unknown_citations(answer, len(sources))
        return RAGAnswer(
            question=selected_question,
            answer=answer,
            sources=sources,
            status=status,
        )

    @staticmethod
    def _source(number: int, passage: RetrievedChunk) -> RAGSource:
        return RAGSource(
            number=number,
            filename=passage.filename,
            page_number=passage.page_number,
            document_id=passage.document_id,
            chunk_id=passage.chunk_id,
            similarity=passage.similarity,
            distance=passage.distance,
            text=passage.text,
        )

    @staticmethod
    def _build_prompt(
        question: str,
        passages: tuple[RetrievedChunk, ...] | list[RetrievedChunk],
        sources: tuple[RAGSource, ...],
    ) -> str:
        evidence = "\n\n".join(
            f"[{source.number}] File: {source.filename}; page: {source.page_number}; "
            f"document_id: {source.document_id}; chunk_id: {source.chunk_id}; "
            f"similarity: {source.similarity if source.similarity is not None else 'unavailable'}\n"
            f"Passage: {passage.text}"
            for source, passage in zip(sources, passages)
        )
        return (
            "You are AgentX, a PDF research assistant. Answer the question using "
            "only the evidence passages below. Do not use outside knowledge or "
            "invent facts, quotations, citations, filenames, pages, or document IDs. "
            "Cite each factual claim with the matching bracketed evidence number, "
            "for example [1]. If the passages do not support an answer, reply with "
            "the exact prefix INSUFFICIENT_EVIDENCE: followed by a brief explanation.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}"
        )

    @staticmethod
    def _remove_unknown_citations(answer: str, source_count: int) -> str:
        """Remove citation numbers not assigned to actual retrieved passages."""
        return _CITATION_PATTERN.sub(
            lambda match: match.group(0)
            if 1 <= int(match.group(1)) <= source_count
            else "",
            answer,
        ).strip()

    @staticmethod
    def _response_text(response: object) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in content
            ).strip()
        return str(content).strip()

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

