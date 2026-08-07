import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

from experimentmind.evidence import MetricType
from experimentmind.statistics import (
    analyze_binary_metric,
    analyze_continuous_metric,
    analyze_experiment,
)
from experimentmind.synthetic import generate_shipping_threshold_experiment


def test_binary_metric_matches_library_reference_example() -> None:
    control = np.array([True] * 20 + [False] * 80)
    treatment = np.array([True] * 30 + [False] * 70)

    result = analyze_binary_metric("converted", control, treatment)
    _, expected_p = proportions_ztest([30, 20], [100, 100])

    assert result.metric_type is MetricType.BINARY
    assert result.control_value == pytest.approx(0.20)
    assert result.treatment_value == pytest.approx(0.30)
    assert result.absolute_effect == pytest.approx(0.10)
    assert result.relative_effect == pytest.approx(0.50)
    assert result.p_value == pytest.approx(expected_p)
    assert result.confidence_interval[0] < result.absolute_effect
    assert result.confidence_interval[1] > result.absolute_effect
    assert result.sample_size_control == 100
    assert result.sample_size_treatment == 100


def test_continuous_metric_matches_welch_reference_example() -> None:
    control = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    treatment = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

    result = analyze_continuous_metric("value", control, treatment)
    reference = stats.ttest_ind(treatment, control, equal_var=False)

    assert result.control_value == pytest.approx(3.0)
    assert result.treatment_value == pytest.approx(6.0)
    assert result.absolute_effect == pytest.approx(3.0)
    assert result.relative_effect == pytest.approx(1.0)
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.confidence_interval[0] < result.absolute_effect
    assert result.confidence_interval[1] > result.absolute_effect


def test_negative_effect_direction_is_preserved() -> None:
    control = np.array([4.0, 5.0, 6.0, 7.0])
    treatment = np.array([1.0, 2.0, 3.0, 4.0])

    result = analyze_continuous_metric("value", control, treatment)

    assert result.absolute_effect < 0
    assert result.relative_effect is not None
    assert result.relative_effect < 0


def test_zero_control_value_has_no_relative_effect() -> None:
    control = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    treatment = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])

    result = analyze_continuous_metric("value", control, treatment)

    assert result.relative_effect is None


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_invalid_alpha_is_rejected(alpha: float) -> None:
    values = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="alpha"):
        analyze_continuous_metric("value", values, values, alpha=alpha)


def test_insufficient_samples_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least two"):
        analyze_binary_metric(
            "converted", np.array([True]), np.array([False, True])
        )


@pytest.mark.parametrize("outcome", [False, True])
def test_binary_metric_with_no_pooled_variance_is_rejected(outcome: bool) -> None:
    control = np.full(100, outcome, dtype=bool)
    treatment = np.full(100, outcome, dtype=bool)

    with pytest.raises(ValueError, match="variance"):
        analyze_binary_metric("converted", control, treatment)


def test_zero_variance_in_both_groups_is_rejected() -> None:
    with pytest.raises(ValueError, match="variance"):
        analyze_continuous_metric("value", np.ones(5), np.ones(5))


def test_synthetic_experiment_builds_complete_evidence() -> None:
    observations = generate_shipping_threshold_experiment(seed=42)
    evidence = analyze_experiment(observations)
    metrics = {metric.metric_name: metric for metric in evidence.metrics}

    assert set(metrics) == {
        "revenue_per_session",
        "conversion_rate",
        "shipping_cost_per_session",
    }
    assert all(metric.sample_size_control == 10_000 for metric in evidence.metrics)
    assert all(metric.sample_size_treatment == 10_000 for metric in evidence.metrics)
    assert all(
        metric.confidence_interval[0]
        <= metric.absolute_effect
        <= metric.confidence_interval[1]
        for metric in evidence.metrics
    )
