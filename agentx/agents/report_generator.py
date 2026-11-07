"""Deterministically assemble an evidence-aware Markdown research report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ReportResult:
    """Markdown report and a coarse report-generation status."""

    markdown: str
    status: str


class ReportGenerator:
    """Build a report from AgentX state without generating new evidence.

    This initial implementation is intentionally deterministic and makes no
    LLM calls. Source references and excerpts are rendered only from the
    preserved ``source_evidence`` records.
    """

    def generate(self, state: Mapping[str, object]) -> ReportResult:
        """Return a Markdown report for a complete or partial workflow state."""
        query = self._string(state.get("user_query"))
        evidence = self._records(state.get("source_evidence"))
        verification = self._verification_records(state.get("verification_results"))
        usable_evidence = [record for record in evidence if self._string(record.get("text"))]
        sections = [
            "# AgentX Research Report",
            "## Original Research Question\n" + (query or "No research question was supplied."),
            self._executive_summary(state, usable_evidence, verification),
            "## Research Objectives\n" + (f"- Address the submitted question: {query}" if query else "- No objective can be derived because the query is missing."),
            self._research_section(state, evidence, verification),
            self._document_section(state, evidence, verification),
            self._verification_section(verification),
            self._problem_claims_section(verification),
            self._limitations_section(state, evidence, verification),
            self._conclusion(state, evidence, verification),
            self._references_section(evidence),
        ]
        markdown = "\n\n".join(section for section in sections if section).strip() + "\n"
        status = "generated" if usable_evidence else "insufficient_evidence"
        return ReportResult(markdown=markdown, status=status)

    def _executive_summary(
        self,
        state: Mapping[str, object],
        evidence: list[dict[str, object]],
        verification: list[dict[str, object]],
    ) -> str:
        parts: list[str] = []
        research_summary = self._string(state.get("research_summary"))
        rag_answer = self._string(state.get("rag_answer"))
        if research_summary:
            parts.append(f"Web research summary: {research_summary}")
        if rag_answer:
            parts.append(f"Document answer: {rag_answer}")
        if not parts:
            parts.append("No research summary or document answer was supplied.")
        supported = sum(item.get("status") == "supported" for item in verification)
        concerns = sum(item.get("status") in {"unsupported", "conflicting", "insufficient_evidence"} for item in verification)
        parts.append(
            f"Evidence review: {supported} finding(s) marked supported and {concerns} finding(s) marked unsupported, conflicting, or insufficient. "
            + (f"{len(evidence)} source passage(s) contain preserved text." if evidence else "No source text was preserved.")
        )
        return "## Executive Summary\n" + "\n\n".join(parts)

    def _research_section(
        self,
        state: Mapping[str, object],
        evidence: list[dict[str, object]],
        verification: list[dict[str, object]],
    ) -> str:
        findings = self._records(state.get("research_results"))
        lines = ["## Web Research Findings"]
        if self._string(state.get("research_summary")):
            lines.append(f"Synthesis: {self._string(state.get('research_summary'))}")
        if not findings:
            lines.append("No web research findings are available.")
        for finding in findings:
            text = self._string(finding.get("text") or finding.get("claim"))
            if not text:
                continue
            status = self._status_for(text, verification)
            citations = self._citation_labels(finding.get("citation_numbers"), evidence, "web")
            suffix = f" References: {', '.join(citations)}." if citations else " No validated source reference is available."
            lines.append(f"- **[{status}]** {text}{suffix}")
        return "\n".join(lines)

    def _document_section(
        self,
        state: Mapping[str, object],
        evidence: list[dict[str, object]],
        verification: list[dict[str, object]],
    ) -> str:
        answer = self._string(state.get("rag_answer"))
        lines = ["## Uploaded Document Findings"]
        if answer:
            lines.append(f"Document QA response (verification status: {self._status_for(answer, verification)}): {answer}")
        else:
            lines.append("No document question-answering result is available.")
        passages = [record for record in evidence if record.get("source_type") == "document"]
        if passages:
            lines.append("\nRetrieved passages preserved from the PDF:")
            for record in passages:
                lines.append(f"- {self._source_label(record)}: {self._quote(self._string(record.get('text')))}")
        elif answer:
            lines.append("No original PDF passage text was preserved; the generated answer alone is not treated as source evidence.")
        return "\n".join(lines)

    def _verification_section(self, verification: list[dict[str, object]]) -> str:
        lines = ["## Evidence Verification"]
        if not verification:
            lines.append("Verification results are unavailable.")
            return "\n".join(lines)
        for item in verification:
            claim = self._string(item.get("claim")) or "Unspecified finding"
            status = self._string(item.get("status")) or "insufficient_evidence"
            explanation = self._string(item.get("explanation"))
            sources = item.get("supporting_sources")
            refs = self._verification_source_labels(sources)
            line = f"- **{status}** — {claim}"
            if refs:
                line += f" (sources: {', '.join(refs)})"
            if explanation:
                line += f". {explanation}"
            contradiction = self._string(item.get("contradiction"))
            if contradiction:
                line += f" Detected contradiction: {contradiction}"
            lines.append(line)
        return "\n".join(lines)

    def _problem_claims_section(self, verification: list[dict[str, object]]) -> str:
        lines = ["## Conflicting or Unsupported Claims"]
        issues = [item for item in verification if item.get("status") in {"unsupported", "conflicting"}]
        if not issues:
            lines.append("No claims were explicitly marked unsupported or conflicting by the verification checks.")
        for item in issues:
            label = self._string(item.get("status"))
            claim = self._string(item.get("claim")) or "Unspecified finding"
            lines.append(f"- **{label}** — {claim}")
        return "\n".join(lines)

    def _limitations_section(
        self,
        state: Mapping[str, object],
        evidence: list[dict[str, object]],
        verification: list[dict[str, object]],
    ) -> str:
        lines = ["## Limitations and Insufficient-Evidence Warnings"]
        warnings: list[str] = []
        if not evidence:
            warnings.append("No source_evidence records were supplied, so this report cannot substantiate findings from original passages.")
        elif not any(self._string(item.get("text")) for item in evidence):
            warnings.append("Source metadata exists, but no source text is available; citations alone do not establish claim support.")
        for item in verification:
            if item.get("status") == "insufficient_evidence":
                claim = self._string(item.get("claim")) or "A finding"
                explanation = self._string(item.get("explanation"))
                warnings.append(f"{claim}: {explanation}" if explanation else f"{claim}: evidence was insufficient.")
        errors = state.get("errors")
        if isinstance(errors, (list, tuple)):
            warnings.extend(f"Workflow issue: {error}" for error in errors if isinstance(error, str) and error.strip())
        if not warnings:
            warnings.append("No additional missing-evidence warning was recorded; verification remains a deterministic lexical check, not proof of semantic entailment.")
        lines.extend(f"- {warning}" for warning in dict.fromkeys(warnings))
        return "\n".join(lines)

    def _conclusion(
        self,
        state: Mapping[str, object],
        evidence: list[dict[str, object]],
        verification: list[dict[str, object]],
    ) -> str:
        if not evidence:
            conclusion = "The available workflow output does not include preserved source passages. Treat generated summaries as unverified and consult the referenced sources when available."
        elif any(item.get("status") in {"conflicting", "unsupported", "insufficient_evidence"} for item in verification):
            conclusion = "The report contains findings that are unsupported, conflicting, or insufficiently evidenced. These findings should not be treated as verified conclusions."
        else:
            conclusion = "This report organizes the supplied findings and preserved evidence. Verification labels reflect the configured deterministic checks and should be interpreted within their stated limitations."
        return "## Conclusion\n" + conclusion

    def _references_section(self, evidence: list[dict[str, object]]) -> str:
        lines = ["## References and Citations"]
        if not evidence:
            lines.append("No source references are available.")
            return "\n".join(lines)
        for record in evidence:
            lines.append(f"- {self._source_label(record)}")
        return "\n".join(lines)

    def _source_label(self, source: Mapping[str, object]) -> str:
        source_type = self._string(source.get("source_type")) or "source"
        citation = source.get("citation_id")
        label = f"{source_type.title()} citation [{citation}]" if citation is not None else f"Source {self._string(source.get('source_id')) or 'reference unavailable'}"
        title = self._string(source.get("source_title") or source.get("filename"))
        if title:
            label += f" — {title}"
        if source_type == "web":
            url = self._string(source.get("source_reference"))
            if url:
                label += f" — {url}"
        else:
            page = source.get("page_number")
            if page is not None:
                label += f", page {page}"
            for key in ("document_id", "chunk_id"):
                value = self._string(source.get(key))
                if value:
                    label += f", {key.replace('_', ' ')} {value}"
        if source.get("similarity") is not None:
            label += f", similarity {source['similarity']}"
        return label

    def _citation_labels(self, numbers: object, evidence: list[dict[str, object]], source_type: str) -> list[str]:
        if not isinstance(numbers, (list, tuple, set)):
            return []
        available = {item.get("citation_id") for item in evidence if item.get("source_type") == source_type}
        return [f"{source_type.title()} [{number}]" for number in numbers if number in available]

    def _status_for(self, claim: str, verification: list[dict[str, object]]) -> str:
        for item in verification:
            if self._string(item.get("claim")) == claim:
                return self._string(item.get("status")) or "verification unavailable"
        return "verification unavailable"

    def _verification_source_labels(self, sources: object) -> list[str]:
        if not isinstance(sources, (list, tuple)):
            return []
        labels: list[str] = []
        for source in sources:
            if not isinstance(source, dict):
                continue
            source_type = self._string(source.get("source_type")) or "source"
            number = source.get("number")
            if number is not None:
                labels.append(f"{source_type.title()} [{number}]")
            elif source.get("source_id"):
                labels.append(f"Source {source['source_id']}")
        return labels

    @staticmethod
    def _quote(text: str) -> str:
        return f"> {text}" if text else "> [No source text was preserved.]"

    @staticmethod
    def _string(value: object) -> str:
        return value.strip() if isinstance(value, str) else ""

    @staticmethod
    def _records(value: object) -> list[dict[str, object]]:
        return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, (list, tuple)) else []

    @classmethod
    def _verification_records(cls, value: object) -> list[dict[str, object]]:
        if isinstance(value, dict) and isinstance(value.get("findings"), (list, tuple)):
            value = value["findings"]
        return cls._records(value)
