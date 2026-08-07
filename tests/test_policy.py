from collections.abc import Iterable

import pytest

from experimentmind.evidence import (
    Evidence,
    EvidenceClassification,
    MetricEvidence,
    MetricRole,
    MetricSpec,
    MetricType,
)
from experimentmind.policy import Decision, recommend


def metric(
    name: str, role: MetricRole, classification: EvidenceClassification
) -> MetricEvidence:
    return MetricEvidence(
        metric_name=name,
        metric_type=MetricType.CONTINUOUS,
        metric_spec=MetricSpec(name, role, True, meaningful_effect=0.10),
        control_value=1.0,
        treatment_value=1.1,
        absolute_effect=0.1,
        relative_effect=0.1,
        confidence_interval=(0.01, 0.19),
        confidence_level=0.95,
        p_value=0.01,
        sample_size_control=100,
        sample_size_treatment=100,
        classification=classification,
    )


def evidence(metrics: Iterable[MetricEvidence]) -> Evidence:
    return Evidence("Policy test", tuple(metrics), alpha=0.05)


POSITIVE = EvidenceClassification.CLEARLY_POSITIVE
NEGATIVE = EvidenceClassification.CLEARLY_NEGATIVE
NEGLIGIBLE = EvidenceClassification.NEGLIGIBLE
UNCERTAIN = EvidenceClassification.UNCERTAIN


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ([metric("primary", MetricRole.PRIMARY, NEGATIVE)], Decision.DO_NOT_SHIP),
        (
            [
                metric("primary", MetricRole.PRIMARY, POSITIVE),
                metric("cost", MetricRole.GUARDRAIL, NEGATIVE),
            ],
            Decision.TRADEOFF,
        ),
        (
            [
                metric("primary", MetricRole.PRIMARY, UNCERTAIN),
                metric("conversion", MetricRole.SECONDARY, POSITIVE),
                metric("cost", MetricRole.GUARDRAIL, NEGATIVE),
            ],
            Decision.TRADEOFF,
        ),
        (
            [
                metric("primary", MetricRole.PRIMARY, UNCERTAIN),
                metric("cost", MetricRole.GUARDRAIL, NEGATIVE),
            ],
            Decision.DO_NOT_SHIP,
        ),
        (
            [
                metric("primary", MetricRole.PRIMARY, POSITIVE),
                metric("cost", MetricRole.GUARDRAIL, UNCERTAIN),
            ],
            Decision.SHIP,
        ),
        ([metric("primary", MetricRole.PRIMARY, NEGLIGIBLE)], Decision.DO_NOT_SHIP),
        ([metric("primary", MetricRole.PRIMARY, UNCERTAIN)], Decision.COLLECT_MORE_DATA),
        (
            [
                metric("primary", MetricRole.PRIMARY, UNCERTAIN),
                metric("secondary", MetricRole.SECONDARY, POSITIVE),
            ],
            Decision.COLLECT_MORE_DATA,
        ),
        (
            [
                metric("primary", MetricRole.PRIMARY, NEGATIVE),
                metric("secondary", MetricRole.SECONDARY, POSITIVE),
                metric("cost", MetricRole.GUARDRAIL, NEGATIVE),
            ],
            Decision.DO_NOT_SHIP,
        ),
    ],
)
def test_policy_rule_precedence(
    metrics: list[MetricEvidence], expected: Decision
) -> None:
    assert recommend(evidence(metrics)).decision is expected


@pytest.mark.parametrize(
    "metrics",
    [
        [metric("secondary", MetricRole.SECONDARY, POSITIVE)],
        [
            metric("primary_one", MetricRole.PRIMARY, POSITIVE),
            metric("primary_two", MetricRole.PRIMARY, POSITIVE),
        ],
    ],
)
def test_policy_requires_exactly_one_primary(
    metrics: list[MetricEvidence],
) -> None:
    with pytest.raises(ValueError, match="exactly one primary"):
        recommend(evidence(metrics))


def test_tradeoff_rationale_names_benefit_and_harmed_guardrail() -> None:
    recommendation = recommend(
        evidence(
            [
                metric("revenue", MetricRole.PRIMARY, UNCERTAIN),
                metric("conversion", MetricRole.SECONDARY, POSITIVE),
                metric("shipping_cost", MetricRole.GUARDRAIL, NEGATIVE),
            ]
        )
    )

    assert recommendation.decision is Decision.TRADEOFF
    assert "conversion" in recommendation.rationale[0]
    assert "shipping_cost" in recommendation.rationale[1]
