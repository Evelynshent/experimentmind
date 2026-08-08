"""Structured competing hypotheses with conservative deterministic checks."""

import json
import re
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .evidence import Evidence, MetricEvidence
from .findings import Direction
from .investigation import InvestigationResult
from .scenarios import ExperimentScenario


class HypothesisStatus(str, Enum):
    """Epistemic relationship between a hypothesis and available evidence."""

    CONSISTENT_WITH_EVIDENCE = "consistent_with_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTRADICTED = "contradicted"


class HypothesisDirectionalClaim(BaseModel):
    """Checkable direction asserted for one evidence reference."""

    model_config = ConfigDict(extra="forbid")

    evidence_ref: str = Field(min_length=1)
    direction: Direction


class Hypothesis(BaseModel):
    """An explicitly unproven explanation for observed evidence."""

    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1)
    supporting_evidence: list[str]
    contradicting_evidence: list[str]
    directional_claims: list[HypothesisDirectionalClaim]
    evidence_needed: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> "Hypothesis":
        all_refs = self.supporting_evidence + self.contradicting_evidence
        if len(set(all_refs)) != len(all_refs):
            raise ValueError("hypothesis evidence references must be unique")
        claim_refs = [claim.evidence_ref for claim in self.directional_claims]
        if len(set(claim_refs)) != len(claim_refs):
            raise ValueError("hypothesis directional claims must be unique")
        if not set(claim_refs).issubset(all_refs):
            raise ValueError("directional claims must cite hypothesis evidence")
        return self


class HypothesisBatch(BaseModel):
    """Two or three competing explanations returned by the optional LLM."""

    model_config = ConfigDict(extra="forbid")

    hypotheses: list[Hypothesis] = Field(min_length=2, max_length=3)


@dataclass(frozen=True)
class VerifiedHypothesis:
    """A hypothesis annotated with a limited epistemic check."""

    hypothesis: Hypothesis
    status: HypothesisStatus
    details: tuple[str, ...]


HYPOTHESIS_INSTRUCTIONS = """Generate 2-3 competing experiment hypotheses.

Use only supplied evidence references. Use tentative language such as may,
might, or could. Copy every asserted evidence direction into directional_claims.
List evidence that would be needed to distinguish the explanation. Hypotheses
are not causal findings and must never be described as verified.
Treat evidence marked post_treatment_conditioned_descriptive as descriptive
only, not as a randomized causal effect.
"""


def evidence_index(
    evidence: Evidence, investigation: InvestigationResult
) -> dict[str, MetricEvidence]:
    """Build stable references for top-line and investigation evidence."""

    indexed = {f"top_line:{metric.metric_name}": metric for metric in evidence.metrics}
    if investigation.segmentation is not None:
        for segment in investigation.segmentation.segments:
            key = (
                f"segment:{segment.dimension}:{segment.segment}:"
                f"{segment.metric.metric_name}"
            )
            indexed[key] = segment.metric
    if investigation.decomposition is not None:
        indexed["decomposition:conversion_rate"] = (
            investigation.decomposition.conversion_rate
        )
        indexed["decomposition:revenue_per_converted_session"] = (
            investigation.decomposition.revenue_per_converted_session
        )
    return indexed


def _direction(metric: MetricEvidence) -> Direction:
    if metric.absolute_effect > 0:
        return Direction.INCREASED
    if metric.absolute_effect < 0:
        return Direction.DECREASED
    return Direction.UNCHANGED


def verify_hypothesis(
    hypothesis: Hypothesis,
    evidence: Evidence,
    investigation: InvestigationResult,
) -> VerifiedHypothesis:
    """Check references and contradictions without claiming causal verification."""

    indexed = evidence_index(evidence, investigation)
    cited = set(hypothesis.supporting_evidence + hypothesis.contradicting_evidence)
    missing = sorted(cited - indexed.keys())
    if missing:
        return VerifiedHypothesis(
            hypothesis,
            HypothesisStatus.INSUFFICIENT_EVIDENCE,
            (f"Evidence does not contain: {', '.join(missing)}.",),
        )

    contradictions = [
        claim.evidence_ref
        for claim in hypothesis.directional_claims
        if claim.direction is not _direction(indexed[claim.evidence_ref])
    ]
    if contradictions or hypothesis.contradicting_evidence:
        references = contradictions or hypothesis.contradicting_evidence
        return VerifiedHypothesis(
            hypothesis,
            HypothesisStatus.CONTRADICTED,
            (
                f"Available evidence contradicts the hypothesis at: {', '.join(references)}.",
            ),
        )

    tentative_terms = ("may", "might", "could", "possibly", "consistent with")
    statement = hypothesis.statement.lower()
    if not hypothesis.supporting_evidence or not any(
        re.search(rf"\b{re.escape(term)}\b", statement) for term in tentative_terms
    ):
        return VerifiedHypothesis(
            hypothesis,
            HypothesisStatus.INSUFFICIENT_EVIDENCE,
            (
                "The hypothesis lacks cited support or appropriately uncertain language.",
            ),
        )
    return VerifiedHypothesis(
        hypothesis,
        HypothesisStatus.CONSISTENT_WITH_EVIDENCE,
        (
            (
                "Cited directions match available evidence; this does not establish "
                "a causal mechanism."
            ),
        ),
    )


def verify_hypotheses(
    hypotheses: list[Hypothesis],
    evidence: Evidence,
    investigation: InvestigationResult,
) -> tuple[VerifiedHypothesis, ...]:
    return tuple(
        verify_hypothesis(hypothesis, evidence, investigation)
        for hypothesis in hypotheses
    )


def generate_hypotheses(
    scenario: ExperimentScenario,
    evidence: Evidence,
    investigation: InvestigationResult,
    *,
    model: str,
    client: OpenAI | None = None,
) -> HypothesisBatch:
    """Generate bounded hypotheses from serialized authoritative evidence."""

    if not model.strip():
        raise ValueError("model must not be empty")
    indexed = evidence_index(evidence, investigation)
    payload = {
        "experiment_name": evidence.experiment_name,
        "experiment_hypothesis": scenario.hypothesis,
        "treatment_description": scenario.treatment_description,
        "evidence": {
            key: {
                "absolute_effect": metric.absolute_effect,
                "confidence_interval": list(metric.confidence_interval),
                "p_value": metric.p_value,
                "classification": (
                    "descriptive_only"
                    if key == "decomposition:revenue_per_converted_session"
                    else metric.classification.value
                ),
                "interpretation_scope": (
                    "post_treatment_conditioned_descriptive"
                    if key == "decomposition:revenue_per_converted_session"
                    else "randomized_intention_to_treat"
                ),
            }
            for key, metric in indexed.items()
        },
    }
    api_client = client if client is not None else OpenAI()
    response = api_client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": HYPOTHESIS_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(payload, indent=2, sort_keys=True),
            },
        ],
        text_format=HypothesisBatch,
    )
    if response.output_parsed is None:
        raise RuntimeError("model response did not contain hypotheses")
    return response.output_parsed
