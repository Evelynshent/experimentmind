from dataclasses import replace

import pytest

from experimentmind.evidence import Evidence, EvidenceClassification
from experimentmind.findings import (
    Direction,
    DirectionalClaim,
    EffectScale,
    Finding,
    FindingType,
    MetricClaim,
)
from experimentmind.statistics import analyze_experiment
from experimentmind.synthetic import generate_shipping_threshold_experiment
from experimentmind.verifier import (
    VerificationStatus,
    verify_finding,
    verify_findings,
)


@pytest.fixture
def evidence() -> Evidence:
    return analyze_experiment(generate_shipping_threshold_experiment(seed=42))


def observation(
    evidence: Evidence,
    metric_name: str = "conversion_rate",
    *,
    effect_scale: EffectScale = EffectScale.RELATIVE,
    effect_value: float | None = None,
    significant: bool | None = None,
    classification: EvidenceClassification | None = None,
) -> Finding:
    metric = next(
        metric for metric in evidence.metrics if metric.metric_name == metric_name
    )
    expected_effect = (
        metric.relative_effect
        if effect_scale is EffectScale.RELATIVE
        else metric.absolute_effect
    )
    assert expected_effect is not None
    return Finding(
        statement=f"Observation about {metric_name}.",
        finding_type=FindingType.OBSERVATION,
        evidence_refs=[metric_name],
        metric_claims=[
            MetricClaim(
                metric_name=metric_name,
                effect_scale=effect_scale,
                effect_value=expected_effect if effect_value is None else effect_value,
                statistically_significant=(metric.p_value < evidence.alpha)
                if significant is None
                else significant,
                classification=metric.classification
                if classification is None
                else classification,
            )
        ],
        directional_claims=[],
        concepts=[],
    )


def interpretation(
    *,
    metric_name: str = "conversion_rate",
    direction: Direction = Direction.INCREASED,
    concepts: list[str] | None = None,
) -> Finding:
    return Finding(
        statement="A tentative interpretation.",
        finding_type=FindingType.INTERPRETATION,
        evidence_refs=[metric_name],
        metric_claims=[],
        directional_claims=[
            DirectionalClaim(metric_name=metric_name, direction=direction)
        ],
        concepts=[metric_name] if concepts is None else concepts,
    )


@pytest.mark.parametrize("effect_scale", [EffectScale.ABSOLUTE, EffectScale.RELATIVE])
def test_correct_observation_is_verified(
    evidence: Evidence, effect_scale: EffectScale
) -> None:
    result = verify_finding(
        observation(evidence, effect_scale=effect_scale), evidence
    )

    assert result.status is VerificationStatus.VERIFIED


def test_rounded_effect_within_tolerance_is_verified(evidence: Evidence) -> None:
    metric = next(
        metric for metric in evidence.metrics if metric.metric_name == "conversion_rate"
    )
    assert metric.relative_effect is not None

    result = verify_finding(
        observation(evidence, effect_value=metric.relative_effect * 1.005), evidence
    )

    assert result.status is VerificationStatus.VERIFIED


@pytest.mark.parametrize(
    "finding",
    [
        lambda evidence: observation(evidence, effect_value=0.50),
        lambda evidence: observation(evidence, significant=False),
        lambda evidence: observation(
            evidence, classification=EvidenceClassification.UNCERTAIN
        ),
    ],
)
def test_incorrect_observation_is_rejected(evidence: Evidence, finding) -> None:
    result = verify_finding(finding(evidence), evidence)

    assert result.status is VerificationStatus.INCORRECT


def test_missing_observation_reference_is_unresolved(evidence: Evidence) -> None:
    finding = Finding(
        statement="Unknown metric changed.",
        finding_type=FindingType.OBSERVATION,
        evidence_refs=["unknown_metric"],
        metric_claims=[
            MetricClaim(
                metric_name="unknown_metric",
                effect_scale=EffectScale.ABSOLUTE,
                effect_value=1.0,
                statistically_significant=True,
                classification=EvidenceClassification.CLEARLY_POSITIVE,
            )
        ],
        directional_claims=[],
        concepts=[],
    )

    assert verify_finding(finding, evidence).status is VerificationStatus.UNRESOLVED


def test_undefined_relative_effect_is_incorrect(evidence: Evidence) -> None:
    metric = evidence.metrics[0]
    zero_control_metric = replace(metric, control_value=0.0, relative_effect=None)
    zero_control_evidence = replace(evidence, metrics=(zero_control_metric,))
    finding = Finding(
        statement="Relative effect exists.",
        finding_type=FindingType.OBSERVATION,
        evidence_refs=[metric.metric_name],
        metric_claims=[
            MetricClaim(
                metric_name=metric.metric_name,
                effect_scale=EffectScale.RELATIVE,
                effect_value=1.0,
                statistically_significant=metric.p_value < evidence.alpha,
                classification=metric.classification,
            )
        ],
        directional_claims=[],
        concepts=[],
    )

    assert (
        verify_finding(finding, zero_control_evidence).status
        is VerificationStatus.INCORRECT
    )


def test_consistent_interpretation_is_labeled_conservatively(evidence: Evidence) -> None:
    result = verify_finding(interpretation(), evidence)

    assert result.status is VerificationStatus.CONSISTENT_WITH_EVIDENCE
    assert "causation is not established" in result.details[0]


def test_directional_contradiction_is_detected(evidence: Evidence) -> None:
    result = verify_finding(
        interpretation(direction=Direction.DECREASED), evidence
    )

    assert result.status is VerificationStatus.CONTRADICTED_BY_EVIDENCE


def test_unobserved_concept_has_insufficient_evidence(evidence: Evidence) -> None:
    result = verify_finding(
        interpretation(concepts=["conversion_rate", "profitability"]), evidence
    )

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE


def test_missing_interpretation_reference_has_insufficient_evidence(
    evidence: Evidence,
) -> None:
    result = verify_finding(
        interpretation(metric_name="unknown_metric", concepts=[]), evidence
    )

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE


def test_contradiction_precedes_unsupported_concept(evidence: Evidence) -> None:
    result = verify_finding(
        interpretation(
            direction=Direction.DECREASED,
            concepts=["conversion_rate", "profitability"],
        ),
        evidence,
    )

    assert result.status is VerificationStatus.CONTRADICTED_BY_EVIDENCE


def test_batch_verification_preserves_order(evidence: Evidence) -> None:
    findings = [observation(evidence), interpretation()]

    results = verify_findings(findings, evidence)

    assert [result.finding for result in results] == findings
    assert [result.status for result in results] == [
        VerificationStatus.VERIFIED,
        VerificationStatus.CONSISTENT_WITH_EVIDENCE,
    ]


@pytest.mark.parametrize("tolerance", [-0.1, float("inf"), float("nan")])
def test_invalid_tolerance_is_rejected(
    evidence: Evidence, tolerance: float
) -> None:
    with pytest.raises(ValueError, match="relative_tolerance"):
        verify_finding(
            observation(evidence), evidence, relative_tolerance=tolerance
        )
