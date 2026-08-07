import pytest

from experimentmind.classification import classify_effect
from experimentmind.evidence import EvidenceClassification, MetricRole, MetricSpec


@pytest.mark.parametrize(
    ("effect", "interval", "p_value", "higher_is_better", "expected"),
    [
        (0.20, (0.10, 0.30), 0.01, True, EvidenceClassification.CLEARLY_POSITIVE),
        (-0.20, (-0.30, -0.10), 0.01, True, EvidenceClassification.CLEARLY_NEGATIVE),
        (-0.20, (-0.30, -0.10), 0.01, False, EvidenceClassification.CLEARLY_POSITIVE),
        (0.20, (0.10, 0.30), 0.01, False, EvidenceClassification.CLEARLY_NEGATIVE),
        (0.05, (0.02, 0.08), 0.01, True, EvidenceClassification.NEGLIGIBLE),
        (0.00, (-0.08, 0.08), 0.50, True, EvidenceClassification.NEGLIGIBLE),
        (0.00, (-0.20, 0.08), 0.50, True, EvidenceClassification.UNCERTAIN),
        (0.20, (0.02, 0.35), 0.10, True, EvidenceClassification.UNCERTAIN),
        (0.10, (0.01, 0.19), 0.01, True, EvidenceClassification.NEGLIGIBLE),
        (-0.10, (-0.19, -0.01), 0.01, True, EvidenceClassification.NEGLIGIBLE),
        (0.00, (-0.10, 0.10), 0.50, True, EvidenceClassification.NEGLIGIBLE),
    ],
)
def test_classification_truth_table(
    effect: float,
    interval: tuple[float, float],
    p_value: float,
    higher_is_better: bool,
    expected: EvidenceClassification,
) -> None:
    result = classify_effect(
        absolute_effect=effect,
        confidence_interval=interval,
        p_value=p_value,
        alpha=0.05,
        metric_spec=MetricSpec(
            "metric", MetricRole.SECONDARY, higher_is_better, meaningful_effect=0.10
        ),
    )

    assert result is expected


@pytest.mark.parametrize(
    ("interval", "threshold", "match"),
    [((1.0, -1.0), 0.1, "lower bound"), ((-1.0, 1.0), -0.1, "non-negative")],
)
def test_invalid_classification_inputs_are_rejected(
    interval: tuple[float, float], threshold: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        classify_effect(
            absolute_effect=0.0,
            confidence_interval=interval,
            p_value=0.5,
            alpha=0.05,
            metric_spec=MetricSpec("metric", MetricRole.SECONDARY, True, threshold),
        )
