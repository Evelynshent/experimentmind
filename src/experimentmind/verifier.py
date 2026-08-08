"""Deterministic verification of structured AI-generated findings."""

import math
from dataclasses import dataclass
from enum import Enum

from .evidence import Evidence, MetricEvidence
from .findings import Direction, EffectScale, Finding, FindingType


class VerificationStatus(str, Enum):
    """Result of checking a structured finding against Evidence."""

    VERIFIED = "verified"
    INCORRECT = "incorrect"
    UNRESOLVED = "unresolved"
    CONSISTENT_WITH_EVIDENCE = "consistent_with_evidence"
    CONTRADICTED_BY_EVIDENCE = "contradicted_by_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass(frozen=True)
class VerifiedFinding:
    """A finding annotated with a deterministic verification result."""

    finding: Finding
    status: VerificationStatus
    details: tuple[str, ...]


def _direction(metric: MetricEvidence) -> Direction:
    if metric.absolute_effect > 0.0:
        return Direction.INCREASED
    if metric.absolute_effect < 0.0:
        return Direction.DECREASED
    return Direction.UNCHANGED


def _verify_observation(
    finding: Finding,
    metrics: dict[str, MetricEvidence],
    alpha: float,
    *,
    relative_tolerance: float,
) -> VerifiedFinding:
    missing = sorted(set(finding.evidence_refs) - metrics.keys())
    if missing:
        return VerifiedFinding(
            finding,
            VerificationStatus.UNRESOLVED,
            (f"Evidence does not contain: {', '.join(missing)}.",),
        )

    errors: list[str] = []
    for claim in finding.metric_claims:
        metric = metrics[claim.metric_name]
        expected_effect = (
            metric.absolute_effect
            if claim.effect_scale is EffectScale.ABSOLUTE
            else metric.relative_effect
        )
        if expected_effect is None:
            errors.append(
                f"{claim.metric_name} has no defined relative effect to verify."
            )
        elif not math.isclose(
            claim.effect_value,
            expected_effect,
            rel_tol=relative_tolerance,
            abs_tol=1e-12,
        ):
            errors.append(
                f"{claim.metric_name} {claim.effect_scale.value} effect does not match Evidence."
            )

        expected_significance = metric.p_value < alpha
        if claim.statistically_significant is not expected_significance:
            errors.append(f"{claim.metric_name} significance does not match Evidence.")
        if claim.classification is not metric.classification:
            errors.append(
                f"{claim.metric_name} classification does not match Evidence."
            )

    if errors:
        return VerifiedFinding(finding, VerificationStatus.INCORRECT, tuple(errors))
    return VerifiedFinding(
        finding,
        VerificationStatus.VERIFIED,
        ("All structured observation claims match Evidence.",),
    )


def _verify_interpretation(
    finding: Finding, metrics: dict[str, MetricEvidence]
) -> VerifiedFinding:
    missing = sorted(set(finding.evidence_refs) - metrics.keys())
    if missing:
        return VerifiedFinding(
            finding,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            (f"Evidence does not contain: {', '.join(missing)}.",),
        )

    contradictions = [
        claim.metric_name
        for claim in finding.directional_claims
        if claim.direction is not _direction(metrics[claim.metric_name])
    ]
    if contradictions:
        return VerifiedFinding(
            finding,
            VerificationStatus.CONTRADICTED_BY_EVIDENCE,
            (
                f"Claimed direction contradicts Evidence for: {', '.join(contradictions)}.",
            ),
        )

    observed_concepts = set(metrics)
    unsupported = sorted(set(finding.concepts) - observed_concepts)
    if unsupported:
        return VerifiedFinding(
            finding,
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            (f"Evidence does not measure concepts: {', '.join(unsupported)}.",),
        )

    return VerifiedFinding(
        finding,
        VerificationStatus.CONSISTENT_WITH_EVIDENCE,
        (
            (
                "Structured interpretation claims do not contradict Evidence; "
                "causation is not established."
            ),
        ),
    )


def verify_finding(
    finding: Finding,
    evidence: Evidence,
    *,
    relative_tolerance: float = 0.01,
) -> VerifiedFinding:
    """Verify one finding against Evidence with a documented 1% tolerance."""

    if relative_tolerance < 0.0 or not math.isfinite(relative_tolerance):
        raise ValueError("relative_tolerance must be finite and non-negative")
    metrics = {metric.metric_name: metric for metric in evidence.metrics}
    if finding.finding_type is FindingType.OBSERVATION:
        return _verify_observation(
            finding,
            metrics,
            evidence.alpha,
            relative_tolerance=relative_tolerance,
        )
    return _verify_interpretation(finding, metrics)


def verify_findings(
    findings: list[Finding], evidence: Evidence
) -> tuple[VerifiedFinding, ...]:
    """Verify findings in input order without mutating them or Evidence."""

    return tuple(verify_finding(finding, evidence) for finding in findings)
