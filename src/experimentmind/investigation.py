"""Evidence sufficiency and bounded planning for V2 investigations."""

import json
from dataclasses import dataclass
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from .analyses import (
    RevenueDecomposition,
    SegmentationResult,
    decompose_revenue,
    segment_metric,
)
from .evidence import Evidence, EvidenceClassification, MetricRole
from .scenarios import ExperimentScenario


class SufficiencyStatus(str, Enum):
    """Whether current evidence can support a decision without investigation."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"


class AnalysisType(str, Enum):
    """The finite analyses that V2 can execute."""

    SEGMENTATION = "segmentation"
    REVENUE_DECOMPOSITION = "revenue_decomposition"


@dataclass(frozen=True)
class SufficiencyAssessment:
    """Deterministic assessment of current top-line evidence."""

    status: SufficiencyStatus
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AnalysisRequest:
    """One scientifically valid, executable next analysis."""

    request_id: str
    analysis_type: AnalysisType
    reason: str
    target_metric: str
    dimension: str | None = None


class PlannerChoice(BaseModel):
    """Schema-constrained LLM ranking among supplied candidates."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


@dataclass(frozen=True)
class InvestigationPlan:
    """Selected analysis and provenance of the bounded choice."""

    selected: AnalysisRequest | None
    selection_reason: str
    selected_by_ai: bool


@dataclass(frozen=True)
class InvestigationResult:
    """Top-line assessment, plan, and deterministic additional evidence."""

    assessment: SufficiencyAssessment
    candidates: tuple[AnalysisRequest, ...]
    plan: InvestigationPlan
    executed_requests: tuple[AnalysisRequest, ...] = ()
    segmentation: SegmentationResult | None = None
    decomposition: RevenueDecomposition | None = None


PLANNER_INSTRUCTIONS = """You rank a finite list of valid experiment analyses.

Choose exactly one request_id from the supplied candidates. Prefer the analysis
most likely to resolve the stated decision uncertainty. Explain the choice
briefly. Do not propose a new metric, dimension, analysis, query, or computation.
"""


def assess_sufficiency(evidence: Evidence) -> SufficiencyAssessment:
    """Assess top-line evidence without using an LLM."""

    primary = [
        metric
        for metric in evidence.metrics
        if metric.metric_spec.role is MetricRole.PRIMARY
    ]
    if len(primary) != 1:
        raise ValueError("sufficiency assessment requires exactly one primary metric")
    primary_metric = primary[0]
    positive_signals = [
        metric.metric_name
        for metric in evidence.metrics
        if metric.metric_spec.role in (MetricRole.PRIMARY, MetricRole.SECONDARY)
        and metric.classification is EvidenceClassification.CLEARLY_POSITIVE
    ]
    harmed_guardrails = [
        metric.metric_name
        for metric in evidence.metrics
        if metric.metric_spec.role is MetricRole.GUARDRAIL
        and metric.classification is EvidenceClassification.CLEARLY_NEGATIVE
    ]

    if primary_metric.classification in (
        EvidenceClassification.CLEARLY_NEGATIVE,
        EvidenceClassification.NEGLIGIBLE,
    ):
        return SufficiencyAssessment(
            SufficiencyStatus.SUFFICIENT,
            (
                (
                    f"Primary metric {primary_metric.metric_name} is "
                    f"{primary_metric.classification.value}; additional analysis "
                    "is not needed to apply the top-line policy."
                ),
            ),
        )
    if positive_signals and harmed_guardrails:
        return SufficiencyAssessment(
            SufficiencyStatus.CONFLICTING,
            (
                f"Positive evidence exists for {', '.join(positive_signals)}.",
                f"Guardrail evidence is negative for {', '.join(harmed_guardrails)}.",
            ),
        )
    if primary_metric.classification is EvidenceClassification.UNCERTAIN:
        return SufficiencyAssessment(
            SufficiencyStatus.INSUFFICIENT,
            (f"Primary metric {primary_metric.metric_name} remains uncertain.",),
        )
    return SufficiencyAssessment(
        SufficiencyStatus.SUFFICIENT,
        (
            (
                f"Primary metric {primary_metric.metric_name} is "
                f"{primary_metric.classification.value} without conflicting evidence."
            ),
        ),
    )


def candidate_analyses(
    scenario: ExperimentScenario,
    evidence: Evidence,
    assessment: SufficiencyAssessment,
) -> tuple[AnalysisRequest, ...]:
    """Return only analyses declared valid before execution."""

    if assessment.status is SufficiencyStatus.SUFFICIENT:
        return ()
    primary = next(
        metric
        for metric in evidence.metrics
        if metric.metric_spec.role is MetricRole.PRIMARY
    )
    candidates: list[AnalysisRequest] = []
    for dimension in scenario.prespecified_dimensions:
        candidates.append(
            AnalysisRequest(
                request_id=f"segment:{primary.metric_name}:{dimension}",
                analysis_type=AnalysisType.SEGMENTATION,
                reason=(
                    f"Test whether the top-line {primary.metric_name} result masks "
                    f"opposing effects across the pre-specified {dimension} groups."
                ),
                target_metric=primary.metric_name,
                dimension=dimension,
            )
        )
    if scenario.supports_revenue_decomposition:
        candidates.append(
            AnalysisRequest(
                request_id="decompose:revenue_per_session",
                analysis_type=AnalysisType.REVENUE_DECOMPOSITION,
                reason=(
                    "Test whether conversion and value per converted session "
                    "offset one another in revenue per session."
                ),
                target_metric="revenue_per_session",
            )
        )
    return tuple(candidates)


def select_analysis(
    candidates: tuple[AnalysisRequest, ...],
    *,
    model: str | None = None,
    client: OpenAI | None = None,
    evidence: Evidence | None = None,
    hypothesis: str | None = None,
) -> InvestigationPlan:
    """Select directly or let an optional LLM rank multiple valid candidates."""

    if not candidates:
        return InvestigationPlan(
            None, "No supported candidate analysis is available.", False
        )
    if len(candidates) == 1:
        return InvestigationPlan(
            candidates[0], "Only one scientifically valid analysis is available.", False
        )
    if model is None:
        return InvestigationPlan(
            candidates[0],
            "Deterministic fallback selected the first pre-specified candidate.",
            False,
        )
    if not model.strip():
        raise ValueError("model must not be empty")

    payload = {
        "experiment_hypothesis": hypothesis,
        "current_evidence": [
            {
                "metric_name": metric.metric_name,
                "role": metric.metric_spec.role.value,
                "absolute_effect": metric.absolute_effect,
                "confidence_interval": list(metric.confidence_interval),
                "p_value": metric.p_value,
                "classification": metric.classification.value,
            }
            for metric in evidence.metrics
        ]
        if evidence is not None
        else [],
        "candidates": [
            {
                "request_id": candidate.request_id,
                "analysis_type": candidate.analysis_type.value,
                "reason": candidate.reason,
                "target_metric": candidate.target_metric,
                "dimension": candidate.dimension,
            }
            for candidate in candidates
        ],
    }
    api_client = client if client is not None else OpenAI()
    response = api_client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": PLANNER_INSTRUCTIONS},
            {
                "role": "user",
                "content": json.dumps(payload, indent=2, sort_keys=True),
            },
        ],
        text_format=PlannerChoice,
    )
    choice = response.output_parsed
    if choice is None:
        raise RuntimeError("model response did not contain a planner choice")
    by_id = {candidate.request_id: candidate for candidate in candidates}
    if choice.request_id not in by_id:
        raise ValueError("planner selected an analysis outside the valid candidates")
    return InvestigationPlan(by_id[choice.request_id], choice.reason, True)


def investigate(
    scenario: ExperimentScenario,
    evidence: Evidence,
    *,
    planner_model: str | None = None,
    client: OpenAI | None = None,
) -> InvestigationResult:
    """Assess and execute each finite candidate at most once when needed."""

    assessment = assess_sufficiency(evidence)
    candidates = candidate_analyses(scenario, evidence, assessment)
    plan = select_analysis(
        candidates,
        model=planner_model,
        client=client,
        evidence=evidence,
        hypothesis=scenario.hypothesis,
    )
    selected = plan.selected
    if selected is None:
        return InvestigationResult(assessment, candidates, plan)
    executed: list[AnalysisRequest] = [selected]
    segmentation: SegmentationResult | None = None
    decomposition: RevenueDecomposition | None = None
    if selected.analysis_type is AnalysisType.SEGMENTATION:
        if selected.dimension is None:
            raise ValueError("segmentation request requires a dimension")
        segmentation = segment_metric(
            scenario,
            metric_name=selected.target_metric,
            dimension=selected.dimension,
            alpha=evidence.alpha,
        )
    else:
        decomposition = decompose_revenue(scenario, alpha=evidence.alpha)

    remaining = [candidate for candidate in candidates if candidate != selected]
    opposing_segments = False
    if segmentation is not None:
        classifications = {
            segment.metric.classification for segment in segmentation.segments
        }
        opposing_segments = {
            EvidenceClassification.CLEARLY_POSITIVE,
            EvidenceClassification.CLEARLY_NEGATIVE,
        }.issubset(classifications)

    if remaining and not opposing_segments:
        follow_up = remaining[0]
        executed.append(follow_up)
        if follow_up.analysis_type is AnalysisType.SEGMENTATION:
            if follow_up.dimension is None:
                raise ValueError("segmentation request requires a dimension")
            segmentation = segment_metric(
                scenario,
                metric_name=follow_up.target_metric,
                dimension=follow_up.dimension,
                alpha=evidence.alpha,
            )
        else:
            decomposition = decompose_revenue(scenario, alpha=evidence.alpha)

    return InvestigationResult(
        assessment,
        candidates,
        plan,
        executed_requests=tuple(executed),
        segmentation=segmentation,
        decomposition=decomposition,
    )
