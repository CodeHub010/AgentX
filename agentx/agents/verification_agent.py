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

_CITATION_PATTERN = re.compile(r"\[((?:\d+\s*,\s*)*\d+)\]")
_TOKEN_PATTERN = re.compile(r"\b\d+(?:\.\d+)?%?|[a-z][a-z0-9]*\b", re.IGNORECASE)
_NUMERIC_VALUE_PATTERN = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(\s*%)?(?![\w.])")
_CONTEXT_WORD_PATTERN = re.compile(r"[a-z][a-z0-9]*", re.IGNORECASE)
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
    "detected": "detect", "detection": "detect", "recognize": "detect",
    "recognizes": "detect",
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
_NONFACTUAL_INTRODUCTIONS = re.compile(
    r"^(?:here (?:is|are)|below (?:is|are)|the following (?:is|are)|"
    r"this (?:section|summary|response) (?:covers|summarizes|describes)|"
    r"based on (?:the )?(?:provided|available) (?:text|information|passages?))\b",
    re.IGNORECASE,
)
_CITATION_ONLY_LINE = re.compile(r"^\s*(?:\[((?:\d+\s*,\s*)*\d+)\]\s*[.,;:]?\s*)+$")
_ABBREVIATIONS = (
    "e.g.", "i.e.", "etc.", "vs.", "approx.", " Dr.", " Mr.", " Mrs.",
    " Ms.", " Prof.", " Fig.", " No.", " Inc.", " Ltd.", " St.",
)
_CNN_TERMS = re.compile(r"\b(?:cnn|convolutional\s+neural\s+networks?)\b", re.IGNORECASE)
_LSTM_TERMS = re.compile(
    r"\b(?:lstm|long\s+short[-\s]+term\s+memory)\b", re.IGNORECASE
)
_MODEL_FLOW = re.compile(
    r"\b(cnn|lstm)\b.{0,100}?\b(?:is\s+)?(?:passed\s+to|pass(?:es)?\s+to|"
    r"fed\s+into|feed(?:s)?\s+into|transferred\s+to|transfer(?:s)?\s+to)\b"
    r".{0,100}?\b(cnn|lstm)\b",
    re.IGNORECASE,
)
_MODEL_FLOW_PHRASES = re.compile(
    r"\b(?:is\s+)?(?:passed\s+to|pass(?:es)?\s+to|fed\s+into|"
    r"feed(?:s)?\s+into|transferred\s+to|transfer(?:s)?\s+to)\b",
    re.IGNORECASE,
)
_MISSING_VALUE_NOTE = re.compile(
    r"\b(?:not\s+(?:specified|provided|reported|available|determined)|"
    r"unavailable|unknown|cannot\s+be\s+determined|could\s+not\s+be\s+determined|"
    r"insufficient\s+information)\b",
    re.IGNORECASE,
)


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
            if not self._is_factual_fragment(claim):
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
            for claim in self._split_claims(rag_answer):
                if rag_status in {"insufficient_evidence", "error"}:
                    reason = (
                        "The RAG Agent reported that retrieved passages were insufficient."
                        if rag_status == "insufficient_evidence"
                        else "The RAG Agent could not produce a verifiable answer because retrieval or generation failed."
                    )
                    results.append(FindingVerification(
                        claim=claim, status="insufficient_evidence",
                        supporting_sources=(), explanation=reason,
                    ))
                else:
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
            (source, evidence_unit)
            for source in matched
            for evidence_unit in cls._evidence_units(cls._source_text(source))
        ]
        if not pairs:
            return FindingVerification(
                claim, "insufficient_evidence", references_out,
                "Citation metadata is valid, but the cited source has no preserved text. A citation's existence alone does not establish support.",
            )

        contradiction = cls._best_contradiction(claim, [unit for _, unit in pairs])
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
            if cls._layer_purpose_has_configuration_only_support(claim, [unit for _, unit in pairs]):
                return FindingVerification(
                    claim, "insufficient_evidence", references_out,
                    "The cited text confirms a related layer or its configuration but does not state the claimed layer purpose or data-flow behavior.",
                )
            return FindingVerification(
                claim, "unsupported", references_out,
                "The cited source text is available but does not meaningfully align with the claim.",
            )
        return FindingVerification(
            claim, "insufficient_evidence", references_out,
            "The cited text overlaps only part of the proposition; the deterministic comparison is inconclusive.",
        )

    @classmethod
    def _layer_purpose_has_configuration_only_support(
        cls, claim: str, sentences: Sequence[str],
    ) -> bool:
        """Avoid calling a layer-purpose claim unsupported when only its config is cited."""
        layer_pattern = re.compile(
            r"\b(input|conv\s*1d|max\s*pool(?:ing)?|dropout|dense|output|lstm)\b",
            re.IGNORECASE,
        )
        claim_layers = {re.sub(r"\s+", "", item).casefold() for item in layer_pattern.findall(claim)}
        if not claim_layers or cls._configuration_facts(claim):
            return False
        # Only apply to a claim about function/flow, not arbitrary missing facts.
        purpose_terms = {
            "reduce", "reduction", "prevent", "avoid", "regularize", "regularization",
            "capture", "extract", "preserve", "learn", "classify", "transform",
            "downsample", "pool", "transfer", "feed", "pass", "receive",
        }
        claim_terms = cls._proposition_terms(claim)
        if not (claim_terms & purpose_terms):
            return False
        for sentence in sentences:
            evidence_layers = {re.sub(r"\s+", "", item).casefold() for item in layer_pattern.findall(sentence)}
            if claim_layers & evidence_layers and cls._configuration_facts(sentence):
                return True
        return False

    @classmethod
    def _best_contradiction(cls, claim: str, sentences: Sequence[str]) -> str | None:
        claim_terms = cls._proposition_terms(claim)
        claim_numeric_facts = cls._numeric_facts(claim)
        claim_negated = cls._negated_predicates(claim)
        for sentence in sentences:
            # A note that a value is absent cannot numerically contradict a
            # concrete value: it supplies no competing value to compare.
            if _MISSING_VALUE_NOTE.search(claim) or _MISSING_VALUE_NOTE.search(sentence):
                continue
            terms = cls._proposition_terms(sentence)
            claim_config = cls._configuration_facts(claim)
            evidence_config = cls._configuration_facts(sentence)
            comparable = cls._comparable_configuration_facts(
                claim_config, evidence_config, sentence
            )
            if not comparable and cls._coverage(claim, sentence) < 0.72:
                continue
            if claim_config or evidence_config:
                # Configuration values are comparable only under the same
                # explicit semantic label. Nearby numbers and table columns
                # do not establish that two values describe the same property.
                for property_name, claim_values, evidence_values in comparable:
                    for claim_value in claim_values:
                        for evidence_value in evidence_values:
                            if claim_value != evidence_value:
                                def display(value: str) -> str:
                                    try:
                                        return str(float(value))
                                    except ValueError:
                                        return value
                                try:
                                    float(claim_value)
                                    float(evidence_value)
                                    description = "different numeric values"
                                except ValueError:
                                    description = "different values"
                                return (
                                    f"The aligned statements give {description} for "
                                    f"the same property ({property_name}: {display(claim_value)} versus "
                                    f"{display(evidence_value)})."
                                )
                evidence_negated = cls._negated_predicates(sentence)
                shared_predicates = claim_terms & terms
                if claim_negated != evidence_negated and shared_predicates:
                    opposite = claim_negated | evidence_negated
                    if opposite & shared_predicates:
                        return "The aligned statements explicitly negate the same predicate."
                continue
            for claim_value, claim_unit, claim_property in claim_numeric_facts:
                for evidence_value, evidence_unit, evidence_property in cls._numeric_facts(sentence):
                    if (claim_value, claim_unit) == (evidence_value, evidence_unit):
                        continue
                    shared = claim_property & evidence_property
                    union = claim_property | evidence_property
                    same_property = (
                        claim_property == evidence_property
                        or (len(shared) >= 2 and len(shared) / len(union) >= 0.5)
                        or (len(claim_property) == len(evidence_property) == 1 and bool(shared))
                    )
                    if same_property:
                        return (
                            "The aligned statements give different numeric values for "
                            f"the same property ({claim_value}{claim_unit} versus "
                            f"{evidence_value}{evidence_unit})."
                        )
            evidence_negated = cls._negated_predicates(sentence)
            shared_predicates = claim_terms & terms
            if claim_negated != evidence_negated and shared_predicates:
                opposite = claim_negated | evidence_negated
                if opposite & shared_predicates:
                    return "The aligned statements explicitly negate the same predicate."
        return None

    @staticmethod
    def _configuration_facts(text: str) -> dict[str, set[str]]:
        """Extract explicitly labeled model configuration values.

        The deliberately narrow parser handles prose, key/value notation, and
        Markdown table rows. It never infers a property's identity from number
        proximity. Tuple-valued input shapes remain a single atomic value.
        """
        facts: dict[str, set[str]] = {}

        def add(
            key: str, value: str, context: str = "", position: int | None = None,
        ) -> None:
            if context:
                locations = [
                    (match.start(), match.group(0).casefold())
                    for match in re.finditer(
                        r"\b(?:dense|output|lstm|conv\s*1d|conv\s*2d|"
                        r"max\s*pool(?:ing)?|average\s*pool(?:ing)?)\b",
                        context, re.IGNORECASE,
                    )
                ]
                if locations:
                    point = position if position is not None else len(context)
                    preceding = [item for item in locations if item[0] <= point]
                    # A value belongs to the most recent layer label before
                    # it. Looking for the nearest label in either direction
                    # incorrectly assigns Dense's value to the next Output row.
                    _, layer = max(preceding, key=lambda item: item[0]) if preceding else min(
                        locations, key=lambda item: item[0]
                    )
                    layer = re.sub(r"\s+", "", layer)
                    if key in {"neurons", "dense_neurons", "output_neurons"}:
                        if layer == "dense":
                            key = "dense_neurons"
                        elif layer == "output":
                            key = "output_neurons"
                        elif key == "dense_neurons":
                            key = "dense_neurons"
                        elif key == "output_neurons":
                            key = "output_neurons"
                        else:
                            key = "neurons"
                    elif key in {"units", "lstm_units", "dense_units", "output_units"}:
                        if layer == "lstm":
                            key = "lstm_units"
                        elif layer == "dense":
                            key = "dense_units"
                        elif layer == "output":
                            key = "output_units"
                    elif key in {"filters", "conv1d_filters", "conv2d_filters"}:
                        if layer == "conv1d":
                            key = "conv1d_filters"
                        elif layer == "conv2d":
                            key = "conv2d_filters"
                    elif key in {"pool_size", "max_pool_size", "average_pool_size"}:
                        if layer in {"maxpool", "maxpooling"}:
                            key = "max_pool_size"
                        elif layer in {"averagepool", "averagepooling"}:
                            key = "average_pool_size"
            normalized = re.sub(r"\s+", "", value).strip(" ,;|")
            if normalized:
                facts.setdefault(key, set()).add(normalized.casefold())

        for raw_line in text.splitlines() or [text]:
            line = raw_line.strip()
            if not line:
                continue
            # input_shape: (1, 78), including common Markdown key/value rows.
            for match in re.finditer(
                r"\binput[ _-]?shape\b\s*(?:(?:is|of)\s+|=|:|\|)?\s*"
                r"(\(\s*\d+(?:\s*,\s*\d+)+\s*\)|\[\s*\d+(?:\s*,\s*\d+)+\s*\])",
                line, re.IGNORECASE,
            ):
                add("input_shape", match.group(1), line, match.start(1))

            for key, pattern in (
                ("filters", r"\bfilters?\b\s*(?:=|:|\|)\s*(\d+(?:\.\d+)?)"),
                # PDF table extraction can discard the separator when a cell
                # wraps onto a new line ("Filters\n64" -> "Filters 64").
                ("filters", r"\bfilters?\b\s+(\d+(?:\.\d+)?)\b"),
                ("filters", r"\b(\d+(?:\.\d+)?)\s+filters?\b"),
                ("kernel_size", r"\bkernel[ _-]?size\b\s*(?:(?:is|of)\s+|=|:|\|)?\s*(\d+(?:\.\d+)?)"),
                ("pool_size", r"\bpool(?:ing)?[ _-]?size\b\s*(?:(?:is|of)\s+|=|:|\|)?\s*(\d+(?:\.\d+)?)"),
                ("dropout_rate", r"\bdropout(?:[ _-]?rate)?\b\s*(?:(?:is|of)\s+|=|:|\|)?\s*(0?\.\d+|\d+(?:\.\d+)?)"),
                ("output_classes", r"\boutput[ _-]?classes\b\s*(?:=|:|\|)\s*(\d+(?:\.\d+)?)"),
                ("batch_size", r"\bbatch\s+size\b\s*(?:=|:)?\s*(\d+(?:\.\d+)?)"),
                ("epochs", r"\bepochs?\b\s*(?:=|:)?\s*(\d+(?:\.\d+)?)"),
                ("training_percentage", r"\btraining(?:\s+data)?\b\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*%"),
                ("testing_percentage", r"\btesting(?:\s+data)?\b\s*(?:=|:)?\s*(\d+(?:\.\d+)?)\s*%"),
                ("training_percentage", r"\b(\d+(?:\.\d+)?)\s*%\s*training\b"),
                ("testing_percentage", r"\b(\d+(?:\.\d+)?)\s*%\s*testing\b"),
                ("output_classes", r"\boutput\b[^.;\n|]{0,35}?\b(\d+(?:\.\d+)?)\s+classes\b"),
                ("neurons", r"\bneurons?\b\s*(?:=|:|\|)\s*(\d+(?:\.\d+)?)"),
                ("neurons", r"\bneurons?\b\s+(\d+(?:\.\d+)?)\b"),
                ("neurons", r"\b(\d+(?:\.\d+)?)\s+neurons?\b"),
                ("learning_rate", r"\blearning[ _-]?rate\b\s*(?:(?:is|of)\s+|=|:|\|)?\s*(\d+(?:\.\d+)?(?:e-\d+)?)"),
                ("optimizer", r"\boptimizer\b\s*(?:is\s+|=|:|\|)?\s*([a-z][a-z0-9_-]*)"),
                ("optimizer", r"\b(adam|sgd|rmsprop|adagrad|adadelta|nadam|adamw)\s+optimizer\b"),
            ):
                for match in re.finditer(pattern, line, re.IGNORECASE):
                    add(key, match.group(1), line, match.start(1))

            # Natural language configuration statements.
            for match in re.finditer(
                r"\b(?:LSTM|long\s+short[- ]term\s+memory)\b"
                r"[^.;\n|]{0,50}?\b(\d+(?:\.\d+)?)\s+units?\b",
                line, re.IGNORECASE,
            ):
                add("lstm_units", match.group(1), line, match.start(1))
            for match in re.finditer(
                r"\b(?:LSTM|long\s+short[- ]term\s+memory)\b"
                r"[^.;\n|]{0,45}?\bunits?\s+(\d+(?:\.\d+)?)\b",
                line, re.IGNORECASE,
            ):
                add("lstm_units", match.group(1), line, match.start(1))
            # Short form "Units: 64" is accepted as a generic units property;
            # it can match LSTM units only when the cited evidence names LSTM.
            for match in re.finditer(r"\bunits?\s*(?:=|:)\s*(\d+(?:\.\d+)?)\b", line, re.IGNORECASE):
                if "lstm_units" not in facts:
                    add("units", match.group(1), line, match.start(1))
            for match in re.finditer(
                r"\b(?:LSTM|long\s+short[- ]term\s+memory)\b"
                r"[^.;\n|]{0,50}?\bunits?\s*(?:=|:)\s*(\d+(?:\.\d+)?)\b",
                line, re.IGNORECASE,
            ):
                add("lstm_units", match.group(1), line, match.start(1))
            for match in re.finditer(
                r"\b(\d+(?:\.\d+)?)\s+units?\b[^.;\n|]{0,35}?"
                r"\b(?:LSTM|long\s+short[- ]term\s+memory)\b",
                line, re.IGNORECASE,
            ):
                add("lstm_units", match.group(1), line, match.start(1))
            # Markdown table rows may put the layer, property and value in
            # separate cells: | LSTM | units | 64 |.
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) >= 3:
                table_properties = {
                    "filters": "filters",
                    "kernel_size": "kernel_size",
                    "pool_size": "pool_size",
                    "dropout_rate": "rate",
                    "dense_neurons": "neurons",
                }
                for index, cell in enumerate(cells[:-1]):
                    for key, label in table_properties.items():
                        if re.fullmatch(label, cell, re.IGNORECASE):
                            value = cells[index + 1].strip(" =:")
                            if re.fullmatch(r"\d+(?:\.\d+)?", value):
                                add(key, value, line, line.find(value))
                    if re.fullmatch(r"rate", cell, re.IGNORECASE) and any(
                        re.search(r"\bDropout\b", c, re.IGNORECASE) for c in cells
                    ):
                        value = cells[index + 1].strip(" =:")
                        if re.fullmatch(r"0?\.\d+|\d+(?:\.\d+)?", value):
                            add("dropout_rate", value, line, line.find(value))
                if any(re.search(r"\bLSTM\b", cell, re.IGNORECASE) for cell in cells):
                    for cell in cells:
                        if re.fullmatch(r"\d+(?:\.\d+)?", cell):
                            if any(re.search(r"\bunits?\b", c, re.IGNORECASE) for c in cells):
                                add("lstm_units", cell, line, line.find(cell))
                if any(re.search(r"\bDense\b", cell, re.IGNORECASE) for cell in cells):
                    for cell in cells:
                        if re.fullmatch(r"\d+(?:\.\d+)?", cell) and any(
                            re.search(r"\b(?:neurons?|units?)\b", c, re.IGNORECASE) for c in cells
                        ):
                            add("dense_neurons", cell, line, line.find(cell))
        return facts

    @classmethod
    def _find_cross_source_contradiction(
        cls, claim: str, document_sources: list[dict[str, object]],
    ) -> tuple[str, list[dict[str, object]]] | None:
        for source in document_sources:
            reason = cls._best_contradiction(claim, cls._evidence_units(cls._source_text(source)))
            if reason:
                return reason, [source]
        return None

    @classmethod
    def _coverage(cls, claim: str, sentence: str) -> float:
        claim_flow = cls._model_flow_signature(claim)
        evidence_flow = cls._model_flow_signature(sentence)
        if claim_flow and evidence_flow and claim_flow != evidence_flow:
            return 0.0
        claim_terms = cls._proposition_terms(claim)
        evidence_terms = cls._proposition_terms(sentence)
        if not claim_terms or not evidence_terms:
            return 0.0
        claim_config = cls._configuration_facts(claim)
        evidence_config = cls._configuration_facts(sentence)
        comparable = cls._comparable_configuration_facts(
            claim_config, evidence_config, sentence
        )
        if claim_config:
            matched_claims: set[str] = set()
            for claim_key, claim_values in claim_config.items():
                evidence_values = evidence_config.get(claim_key)
                if evidence_values and claim_values & evidence_values:
                    matched_claims.add(claim_key)
            if len(matched_claims) == len(claim_config):
                return 1.0
            # Explicit configuration claims must match the same labeled
            # property and value. Surface-word overlap alone is not enough.
            if not comparable:
                return 0.0
        overlap = len(claim_terms & evidence_terms)
        coverage = overlap / len(claim_terms)
        return min(coverage, 0.77) if claim_config else coverage

    @staticmethod
    def _comparable_configuration_facts(
        claim_facts: dict[str, set[str]],
        evidence_facts: dict[str, set[str]],
        evidence_text: str,
    ) -> list[tuple[str, set[str], set[str]]]:
        pairs = [
            (key, claim_facts[key], evidence_facts[key])
            for key in claim_facts.keys() & evidence_facts.keys()
        ]
        return pairs

    @classmethod
    def _evidence_units(cls, text: str) -> list[str]:
        """Return individual sentences and bounded adjacent-sentence pairs."""
        sentences = cls._sentences(text)
        return sentences + [
            f"{left} {right}" for left, right in zip(sentences, sentences[1:])
        ]

    @staticmethod
    def _model_flow_signature(text: str) -> tuple[str, str] | None:
        """Return direction for explicit CNN/LSTM data-flow language."""
        normalized = _CNN_TERMS.sub("CNN", text)
        normalized = _LSTM_TERMS.sub("LSTM", normalized)
        match = _MODEL_FLOW.search(normalized)
        return (match.group(1).upper(), match.group(2).upper()) if match else None

    @classmethod
    def _proposition_terms(cls, text: str) -> set[str]:
        text = _CITATION_PATTERN.sub("", text)
        normalized = _CNN_TERMS.sub(" cnn ", text)
        normalized = _LSTM_TERMS.sub(" lstm ", normalized)
        normalized = _MODEL_FLOW_PHRASES.sub(" modeltransfer ", normalized)
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

    @classmethod
    def _numeric_facts(
        cls, text: str,
    ) -> list[tuple[float, str, frozenset[str]]]:
        """Extract numbers with nearby property terms, excluding citation metadata."""
        normalized = _CITATION_PATTERN.sub(" ", text)
        normalized = re.sub(
            r"\(?\s*(?:page|p\.)\s*\d+\s*\)?", " ", normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\b(?:document|doc|chunk)[\s_-]*id\s*[:=]?\s*[^\s,;.)\]]+",
            " ", normalized, flags=re.IGNORECASE,
        )
        normalized = re.sub(r"\b(?:doc|chunk)-[a-z0-9_.:-]+\b", " ", normalized, flags=re.IGNORECASE)

        words = list(_CONTEXT_WORD_PATTERN.finditer(normalized))
        facts: list[tuple[float, str, frozenset[str]]] = []
        for match in _NUMERIC_VALUE_PATTERN.finditer(normalized):
            before = [word.group(0) for word in words if word.end() <= match.start()][-3:]
            after = [word.group(0) for word in words if word.start() >= match.end()][:3]
            property_terms = cls._proposition_terms(" ".join(before + after))
            if property_terms:
                facts.append((float(match.group(1)), "%" if match.group(2) else "", frozenset(property_terms)))
        return facts

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
            for citation in _CITATION_PATTERN.findall(text):
                numbers.extend(int(number.strip()) for number in citation.split(","))
        return tuple(dict.fromkeys(numbers))

    @staticmethod
    def _split_claims(answer: str) -> list[str]:
        # Keep Markdown rows whole and carry explicit layer headings into
        # following configuration lines. Headings are context, not claims.
        logical_lines: list[str] = []
        layer_context = ""
        for raw_line in answer.strip().splitlines():
            line = raw_line.strip()
            if not line or re.fullmatch(r"(?:[-*_]\s*){3,}", line):
                continue
            heading = re.sub(r"^(?:[-*•]\s*)", "", line)
            heading = re.sub(r"^#{1,6}\s*", "", heading).strip(" *_")
            layer = VerificationAgent._explicit_layer_heading(heading)
            if layer:
                layer_context = layer
                continue
            if re.match(r"^#{1,6}\s+", line):
                layer_context = ""
                continue
            if _CITATION_ONLY_LINE.fullmatch(line) and logical_lines:
                logical_lines[-1] = f"{logical_lines[-1].rstrip()} {line}"
                continue
            if line.startswith("|") and re.fullmatch(
                r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?", line
            ):
                continue
            if (line.endswith(":") and len(line.split()) <= 8
                    and not _CITATION_PATTERN.search(line)
                    and not re.match(
                        r"(?i)^(?:input\s*shape|units?|filters?|kernel\s*size|pool(?:ing)?\s*size|"
                        r"dropout(?:\s*rate)?|rate|neurons?|classes|batch\s*size|epochs?|"
                        r"optimizer|learning\s*rate|activation)\s*:", line,
                    )
                    and not re.match(r"^(?:[-*•]|\d+[.)])\s+", line)):
                layer_context = ""
                continue
            if logical_lines and re.search(
                r"(?:[:=]|\b(?:units?|filters?|neurons?|rate|size))\s*$",
                logical_lines[-1], re.IGNORECASE,
            ):
                logical_lines[-1] = f"{logical_lines[-1].rstrip()} {line}"
                continue
            if layer_context:
                content = re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", line).strip()
                line = f"{layer_context} — {content}"
            logical_lines.append(line)

        protected: dict[str, str] = {}

        def protect(match: re.Match[str]) -> str:
            marker = f"__AGENTX_DOT_{len(protected)}__"
            protected[marker] = match.group(0)
            return marker

        fragments: list[str] = []
        for line in logical_lines:
            if "|" in line:
                fragments.append(line)
                continue
            text = re.sub(r"(?<=\d)\.(?=\d)", protect, line)
            for abbreviation in _ABBREVIATIONS:
                text = re.sub(re.escape(abbreviation), protect, text, flags=re.IGNORECASE)
            parts = re.split(r"(?<=[.!?])\s+(?!\[(?:\d+\s*,\s*)*\d+\])", text)
            fragments.extend(
                re.sub(r"__AGENTX_DOT_\d+__", lambda match: protected[match.group(0)], part)
                for part in parts
            )
        claims = [re.sub(r"^(?:[-*•]\s*|\d+[.)]\s*)", "", fragment).strip() for fragment in fragments]
        return [claim for claim in claims if VerificationAgent._is_factual_fragment(claim)]

    @staticmethod
    def _explicit_layer_heading(text: str) -> str:
        """Recognize only standalone, explicit layer headings."""
        cleaned = re.sub(r"\s+\[[0-9,\s]+\]\s*$", "", text).strip(" :—–-*")
        for pattern, canonical in (
            (r"input", "Input Layer"), (r"conv\s*1d", "Conv1D Layer"),
            (r"max\s*pool(?:ing)?", "Max Pooling Layer"),
            (r"dropout", "Dropout Layer"), (r"dense", "Dense Layer"),
            (r"output", "Output Layer"), (r"lstm", "LSTM Layer"),
        ):
            if re.fullmatch(rf"(?:Layer\s+\d+\s*[:.—-]\s*)?{pattern}(?:\s+layer)?", cleaned, re.IGNORECASE):
                return canonical
            if re.fullmatch(rf"Layer\s+\d+\s*[:.—-]\s*{pattern}(?:\s+layer)?", cleaned, re.IGNORECASE):
                return canonical
        return ""

    @staticmethod
    def _is_factual_fragment(fragment: str) -> bool:
        """Discard structural labels and obvious non-assertive introductions."""
        text = fragment.strip()
        if re.fullmatch(r"(?:[-*_]\s*){3,}", text) or re.match(r"^#{1,6}\s+", text):
            return False
        if "|" in text:
            cells = [cell.strip() for cell in text.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
                return False
            headers = {"layer", "type", "property", "value", "parameter", "configuration", "setting", "component", "citation", "source"}
            if cells and all(cell.casefold() in headers for cell in cells):
                return False
        text = re.sub(r"^\*{1,2}|\*{1,2}$|^_{1,2}|_{1,2}$", "", text).strip()
        if not text or _NONFACTUAL_INTRODUCTIONS.match(text):
            return False
        if text.endswith(":") and len(text.split()) <= 10 and not _CITATION_PATTERN.search(text):
            return False
        if len(text.split()) <= 8 and re.fullmatch(
            r"(?i)(executive summary|research objectives?|synthesis|(?:uploaded )?document findings?|"
            r"web research findings?|evidence verification|references and citations?|conclusion|"
            r"limitations(?: and insufficient-evidence warnings)?)", text
        ):
            return False
        return True

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
