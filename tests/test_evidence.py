from dataclasses import FrozenInstanceError

import pytest

from experimentmind.evidence import (
    Evidence,
    EvidenceClassification,
    MetricEvidence,
    MetricSpec,
    MetricType,
)


def sample_metric() -> MetricEvidence:
    return MetricEvidence(
        metric_name="conversion_rate",
        metric_type=MetricType.BINARY,
        metric_spec=MetricSpec("conversion_rate", True, 0.002),
        control_value=0.10,
        treatment_value=0.12,
        absolute_effect=0.02,
        relative_effect=0.20,
        confidence_interval=(0.001, 0.039),
        confidence_level=0.95,
        p_value=0.04,
        sample_size_control=1_000,
        sample_size_treatment=1_000,
        classification=EvidenceClassification.CLEARLY_POSITIVE,
    )


def test_evidence_construction_preserves_metric_and_metadata() -> None:
    metric = sample_metric()
    evidence = Evidence("Threshold test", (metric,), alpha=0.05)

    assert evidence.experiment_name == "Threshold test"
    assert evidence.metrics == (metric,)
    assert evidence.alpha == 0.05


def test_evidence_and_metric_are_immutable() -> None:
    metric = sample_metric()
    evidence = Evidence("Threshold test", (metric,), alpha=0.05)

    with pytest.raises(FrozenInstanceError):
        metric.p_value = 0.50  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        evidence.alpha = 0.10  # type: ignore[misc]
