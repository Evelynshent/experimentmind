"""Deterministic decision policy for classified experiment evidence."""

from dataclasses import dataclass
from enum import Enum

from .analyses import SegmentationResult
from .evidence import Evidence, EvidenceClassification, MetricRole


class Decision(str, Enum):
    """Product action returned by the deterministic policy."""

    SHIP = "ship"
    DO_NOT_SHIP = "do_not_ship"
    COLLECT_MORE_DATA = "collect_more_data"
    TRADEOFF = "tradeoff"
    VALIDATE_HETEROGENEITY = "validate_heterogeneity"


@dataclass(frozen=True)
class Recommendation:
    """A policy decision and the evidence-based rules that produced it."""

    decision: Decision
    rationale: tuple[str, ...]


def recommend(evidence: Evidence) -> Recommendation:
    """Map classified evidence to a recommendation using explicit precedence."""

    primary = [
        metric
        for metric in evidence.metrics
        if metric.metric_spec.role is MetricRole.PRIMARY
    ]
    if len(primary) != 1:
        raise ValueError("decision policy requires exactly one primary metric")

    primary_metric = primary[0]
    negative_guardrails = [
        metric
        for metric in evidence.metrics
        if metric.metric_spec.role is MetricRole.GUARDRAIL
        and metric.classification is EvidenceClassification.CLEARLY_NEGATIVE
    ]
    positive_signals = [
        metric
        for metric in evidence.metrics
        if metric.metric_spec.role in (MetricRole.PRIMARY, MetricRole.SECONDARY)
        and metric.classification is EvidenceClassification.CLEARLY_POSITIVE
    ]

    primary_name = primary_metric.metric_name
    primary_classification = primary_metric.classification
    if primary_classification is EvidenceClassification.CLEARLY_NEGATIVE:
        return Recommendation(
            Decision.DO_NOT_SHIP,
            (f"Primary metric {primary_name} is clearly negative.",),
        )

    if negative_guardrails and positive_signals:
        harmed = ", ".join(metric.metric_name for metric in negative_guardrails)
        benefits = ", ".join(metric.metric_name for metric in positive_signals)
        return Recommendation(
            Decision.TRADEOFF,
            (
                f"Positive evidence exists for {benefits}.",
                f"Guardrail evidence is clearly negative for {harmed}.",
            ),
        )

    if negative_guardrails:
        harmed = ", ".join(metric.metric_name for metric in negative_guardrails)
        return Recommendation(
            Decision.DO_NOT_SHIP,
            (f"Guardrail evidence is clearly negative for {harmed}.",),
        )

    if primary_classification is EvidenceClassification.CLEARLY_POSITIVE:
        return Recommendation(
            Decision.SHIP,
            (
                f"Primary metric {primary_name} is clearly positive with no harmed guardrail.",
            ),
        )

    if primary_classification is EvidenceClassification.NEGLIGIBLE:
        return Recommendation(
            Decision.DO_NOT_SHIP,
            (f"Primary metric {primary_name} has no meaningful effect.",),
        )

    return Recommendation(
        Decision.COLLECT_MORE_DATA,
        (f"Primary metric {primary_name} remains uncertain.",),
    )


def recommend_after_investigation(
    evidence: Evidence,
    *,
    segmentation: SegmentationResult | None = None,
) -> Recommendation:
    """Extend V1 policy only when material opposing segment effects exist."""

    if segmentation is not None:
        primary = [
            metric
            for metric in evidence.metrics
            if metric.metric_spec.role is MetricRole.PRIMARY
        ]
        if len(primary) != 1:
            raise ValueError("decision policy requires exactly one primary metric")
        if segmentation.metric_name != primary[0].metric_name:
            raise ValueError(
                "heterogeneity policy requires primary-metric segmentation"
            )
        classifications = {
            segment.metric.classification for segment in segmentation.segments
        }
        if {
            EvidenceClassification.CLEARLY_POSITIVE,
            EvidenceClassification.CLEARLY_NEGATIVE,
        }.issubset(classifications):
            return Recommendation(
                Decision.VALIDATE_HETEROGENEITY,
                (
                    (
                        f"Pre-specified {segmentation.dimension} segments have opposing "
                        f"material effects for {segmentation.metric_name}."
                    ),
                    "Confirm the interaction before a global rollout or targeted treatment.",
                ),
            )
    return recommend(evidence)
