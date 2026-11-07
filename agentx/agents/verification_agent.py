"""Conservative, deterministic checks for claims and their cited evidence.

The verifier uses sentence-level lexical alignment plus a small, explicit set
of wording equivalences. It is not an entailment model or semantic fact-checker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence


VerificationStatus = Literal[
    "supported", "unsupported", "conflicting", "insufficient_evidence"
]

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")
_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?|[a-z][a-z0-9]*\b", re.IGNORECASE)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|[;\n]+")
_NEGATION_WORDS = {"not", "no", "never", "cannot", "cant", "without"}
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "for", "from", "had", "has", "have", "in", "into", "is", "it", "its",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were",
    "what", "which", "with", "using", "uses", "used", "can", "could", "does",
    "do", "did", "would", "should", "will", "a", "an",
}

# Deliberately small and auditable: equivalence is limited to common
# inflections and close wording variants, not arbitrary semantic similarity.
_ALIASES = {
    "convolutional": "cnn", "convolutions": "cnn", "network": "network",
    "networks": "network", "extracts": "extract", "extracted": "extract",
    "extracting": "extract", "captures": "extract", "captured": "extract",
    "capture": "extract", "derive": "extract", "derives": "extract",
    "derived": "extract", "features": "feature", "characteristics": "feature",
    "attributes": "feature", "patterns": "feature", "identifies": "detect",
    "identified": "detect", "identify": "detect", "detects": "detect",
    "detected": "detect", "detection": "detect", "recognizes": "detect",
    "recognised": "detect", "recognizes": "detect", "employs": "use",
    "utilizes": "use", "utilises": "use", "utilize": "use", "utilise": "use",
    "improves": "improve", "improved": "improve", "enhances": "improve",
    "enhanced": "improve", "boosts": "improve", "reduces": "reduce",
    "reduced": "reduce", "decreases": "reduce", "decreased": "reduce",
    "increases": "increase", "increased": "increase", "rises": "increase",
    "rose": "increase", "falls": "decrease", "fell": "decrease",
    "sequences": "sequence", "sequential": "sequence", "temporal": "time",
    "time": "time", "spatial": "space", "models": "model", "systems": "system",
    "approach": "method", "approaches": "method", "methods": "method",
    "methodology": "method",
}
_AUXILIARIES = {"do", "does", "did", "is", "are", "was", "were", "has", "have", "had", "can", "could", "will", "would", "should"}


@dataclass(frozen=True, slots=True)
class FindingVerification:
    """Verification assessment for one claim or finding."""

    claim: str
    status: VerificationStatus
    supporting_sources: tuple[dict[str, object], ...]
    explanation: str
    contradiction: str | None = None


@dataclass(frozen=True, slots=True)
class VerificationReport:
    """All finding-level checks and a concise overall status."""

    status: VerificationStatus
    findings: tuple[FindingVerification, ...]


class VerificationAgent:
    """Validate citation IDs, then compare each claim with cited source text.

    Missing citations or source text yield ``insufficient_evidence``. A valid
    citation is not proof. ``unsupported`` means usable cited text was
    available but did not align with the claim; ``conflicting`` requires a
    clearly aligned opposite proposition or incompatible numeric value.
    """

    def verify(self, state: Mapping[str, object]) -> VerificationReport:
        """Verify web findings and document answer claims from shared state."""
        web_sources = self._sources(state, "web", "research_citations")
        document_sources = self._sources(state, "document", "retrieved_context")
        results: list[FindingVerification] = []

        for item in self._records(state.get("research_results")):
            claim = str(item.get("text") or item.get("claim") or "").strip()
            if not claim:
                continue
            references = self._reference_numbers(item.get("citation_numbers"), claim)
            result = self._verify_claim(claim, references, web_sources, source_type="web")
            # Cross-check a web claim against PDF text only when the web claim
            # itself has a valid citation and actual cited text.
            if references and result.status in {"supported", "conflicting"}:
                disagreement = self._find_cross_source_contradiction(claim, document_sources)
                if disagreement:
                    result = FindingVerification(
                        claim=claim,
                        status="conflicting",
                        supporting_sources=tuple(
                            self._source_reference("web", source)
                            for source in self._matching_sources(references, web_sources)
                        ) + tuple(
                            self._source_reference("document", source)
                            for source in disagreement[1]
                        ),
                        explanation="The cited web text and an available PDF passage align to the same proposition but disagree on its polarity or value.",
                        contradiction=disagreement[0],
                    )
            results.append(result)

        rag_answer = state.get("rag_answer")
        if isinstance(rag_answer, str) and rag_answer.strip():
            rag_status = state.get("rag_status")
            if rag_status in {"insufficient_evidence", "error"}:
                reason = (
                    "The RAG Agent reported that retrieved passages were insufficient."
                    if rag_status == "insufficient_evidence"
                    else "The RAG Agent could not produce a verifiable answer because retrieval or generation failed."
                )
                results.append(FindingVerification(
                    claim=rag_answer.strip(), status="insufficient_evidence",
                    supporting_sources=(), explanation=reason,
                ))
            else:
                for claim in self._split_claims(rag_answer):
                    references = self._reference_numbers(None, claim)
                    results.append(self._verify_claim(
                        claim, references, document_sources, source_type="document"
                    ))

        if not results:
            results.append(FindingVerification(
                claim="No research finding or document answer was available to verify.",
                status="insufficient_evidence", supporting_sources=(),
                explanation="The workflow state contains no claim text to assess.",
            ))
        return VerificationReport(self._overall_status(results), tuple(results))

    @classmethod
    def _verify_claim(
        cls, claim: str, references: Sequence[int], sources: list[dict[str, object]],
        *, source_type: str,
    ) -> FindingVerification:
        if not references:
            return FindingVerification(
                claim, "insufficient_evidence", (),
                "The claim has no citation reference, so its factual support cannot be assessed.",
            )

        by_number = {
            number: source for source in sources
            if (number := cls._number(source.get("number"))) is not None
        }
        invalid = [number for number in references if number not in by_number]
        matched = [by_number[number] for number in references if number in by_number]
        references_out = tuple(cls._source_reference(source_type, source) for source in matched)
        if invalid:
            return FindingVerification(
                claim, "unsupported", references_out,
                f"Citation reference(s) {invalid} do not correspond to available {source_type} source metadata.",
            )

        pairs = [
            (source, sentence)
            for source in matched
            for sentence in cls._sentences(cls._source_text(source))
        ]
        if not pairs:
            return FindingVerification(
                claim, "insufficient_evidence", references_out,
                "Citation metadata is valid, but the cited source has no preserved text. A citation's existence alone does not establish support.",
            )

        contradiction = cls._best_contradiction(claim, [sentence for _, sentence in pairs])
        if contradiction:
            return FindingVerification(
                claim, "conflicting", references_out,
                "A cited sentence aligns with the same subject and proposition but states an incompatible polarity or numeric value.",
                contradiction=contradiction,
            )

        best_coverage = max((cls._coverage(claim, sentence) for _, sentence in pairs), default=0.0)
        if best_coverage >= 0.78:
            return FindingVerification(
                claim, "supported", references_out,
                "Cited sentence meaningfully aligns with the claim after conservative wording normalization; this is not a guarantee of entailment or factual truth.",
            )
        if best_coverage < 0.35:
            return FindingVerification(
                claim, "unsupported", references_out,
                "The cited source text is available but does not meaningfully align with the claim.",
            )
        return FindingVerification(
            claim, "insufficient_evidence", references_out,
            "The cited text overlaps only part of the proposition; the deterministic comparison is inconclusive.",
        )

    @classmethod
    def _best_contradiction(cls, claim: str, sentences: Sequence[str]) -> str | None:
        claim_terms = cls._proposition_terms(claim)
        claim_numbers = cls._numbers(claim)
        claim_negated = cls._negated_predicates(claim)
        for sentence in sentences:
            terms = cls._proposition_terms(sentence)
            if cls._coverage(claim, sentence) < 0.72:
                continue
            evidence_numbers = cls._numbers(sentence)
            if claim_numbers and evidence_numbers and claim_numbers.isdisjoint(evidence_numbers):
                return f"The aligned statements give different numeric values ({sorted(claim_numbers)} versus {sorted(evidence_numbers)})."
            evidence_negated = cls._negated_predicates(sentence)
            shared_predicates = claim_terms & terms
            if claim_negated != evidence_negated and shared_predicates:
                opposite = claim_negated | evidence_negated
                if opposite & shared_predicates:
                    return "The aligned statements explicitly negate the same predicate."
        return None

    @classmethod
    def _find_cross_source_contradiction(
        cls, claim: str, document_sources: list[dict[str, object]],
    ) -> tuple[str, list[dict[str, object]]] | None:
        for source in document_sources:
            reason = cls._best_contradiction(claim, cls._sentences(cls._source_text(source)))
            if reason:
                return reason, [source]
        return None

    @classmethod
    def _coverage(cls, claim: str, sentence: str) -> float:
        claim_terms = cls._proposition_terms(claim)
        evidence_terms = cls._proposition_terms(sentence)
        if not claim_terms or not evidence_terms:
            return 0.0
        overlap = len(claim_terms & evidence_terms)
        return overlap / len(claim_terms)

    @classmethod
    def _proposition_terms(cls, text: str) -> set[str]:
        text = _CITATION_PATTERN.sub("", text)
        normalized = re.sub(
            r"\bconvolutional\s+neural\s+networks?\b", " cnn ", text,
            flags=re.IGNORECASE,
        )
        result: set[str] = set()
        for raw in _TOKEN_PATTERN.findall(normalized):
            token = raw.casefold()
            if token in _STOP_WORDS or token in _NEGATION_WORDS or re.fullmatch(r"\d+(?:\.\d+)?%?", token):
                continue
            token = _ALIASES.get(token, token)
            # Regular inflection normalization for common English plurals.
            if token.endswith("ies") and len(token) > 4:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 3 and token not in {"news", "analysis"}:
                token = token[:-1]
            result.add(token)
        return result

    @classmethod
    def _negated_predicates(cls, text: str) -> set[str]:
        """Return only predicates locally governed by explicit negation."""
        normalized = re.sub(r"\s+", " ", text.casefold())
        patterns = (
            r"\b(?:do|does|did|is|are|was|were|has|have|had|can|could|will|would|should)?\s*not\s+(?:\w+ly\s+)?([a-z]+)",
            r"\bnever\s+(?:\w+ly\s+)?([a-z]+)",
            r"\bcannot\s+([a-z]+)",
        )
        predicates: set[str] = set()
        for pattern in patterns:
            for match in re.finditer(pattern, normalized):
                raw = match.group(1)
                predicates.add(_ALIASES.get(raw, raw))
        return predicates

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [part.strip() for part in _SENTENCE_PATTERN.split(text) if part.strip()]

    @staticmethod
    def _numbers(text: str) -> set[str]:
        text = _CITATION_PATTERN.sub("", text)
        return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", text))

    @staticmethod
    def _source_text(source: Mapping[str, object]) -> str:
        for key in ("evidence_text", "snippet", "text", "evidence", "content"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

    @classmethod
    def _source_reference(cls, source_type: str, source: Mapping[str, object]) -> dict[str, object]:
        reference: dict[str, object] = {
            "source_type": source_type,
            "number": cls._number(source.get("number")),
        }
        keys = (
            "source_id", "source_reference", "source_title", "title", "url",
            "published_date", "filename", "page_number", "document_id", "chunk_id", "similarity",
        )
        reference.update({key: source[key] for key in keys if key in source})
        return reference

    @classmethod
    def _matching_sources(cls, references: Sequence[int], sources: list[dict[str, object]]) -> list[dict[str, object]]:
        selected = set(references)
        return [source for source in sources if cls._number(source.get("number")) in selected]

    @classmethod
    def _sources(cls, state: Mapping[str, object], source_type: str, fallback_key: str) -> list[dict[str, object]]:
        """Merge preserved source records with agent citations by citation ID."""
        preserved: list[dict[str, object]] = []
        for evidence in cls._records(state.get("source_evidence")):
            if evidence.get("source_type") != source_type:
                continue
            source = dict(evidence)
            source["number"] = evidence.get("citation_id")
            source["text"] = evidence.get("text")
            if source_type == "web":
                source["title"] = evidence.get("source_title")
                source["url"] = evidence.get("source_reference")
                source["snippet"] = evidence.get("text")
            else:
                source["filename"] = evidence.get("filename") or evidence.get("source_title")
            preserved.append(source)

        by_number = {cls._number(item.get("number")): item for item in preserved if cls._number(item.get("number")) is not None}
        for raw in cls._records(state.get(fallback_key)):
            number = cls._number(raw.get("number"))
            if number is not None and number in by_number:
                merged = by_number[number]
                for key, value in raw.items():
                    if key not in merged or merged[key] in (None, ""):
                        merged[key] = value
            else:
                preserved.append(dict(raw))
        return preserved

    @staticmethod
    def _records(value: object) -> list[dict[str, object]]:
        if not isinstance(value, (list, tuple)):
            return []
        return [item for item in value if isinstance(item, dict)]

    @staticmethod
    def _number(value: object) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int) and value > 0:
            return value
        return None

    @staticmethod
    def _reference_numbers(value: object, text: str) -> tuple[int, ...]:
        numbers: list[int] = []
        if isinstance(value, (list, tuple, set)):
            numbers.extend(item for item in value if isinstance(item, int) and not isinstance(item, bool) and item > 0)
        if not numbers:
            numbers.extend(int(number) for number in _CITATION_PATTERN.findall(text))
        return tuple(dict.fromkeys(numbers))

    @staticmethod
    def _split_claims(answer: str) -> list[str]:
        fragments = re.split(r"(?<=[.!?])\s+|\r?\n+", answer.strip())
        claims = [re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", fragment).strip() for fragment in fragments]
        return [claim for claim in claims if claim]

    @staticmethod
    def _overall_status(results: Sequence[FindingVerification]) -> VerificationStatus:
        statuses = {result.status for result in results}
        if "conflicting" in statuses:
            return "conflicting"
        if "unsupported" in statuses:
            return "unsupported"
        if statuses == {"supported"}:
            return "supported"
        return "insufficient_evidence"
