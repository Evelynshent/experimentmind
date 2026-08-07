"""Deterministic classification of statistical experiment evidence."""

import math

from .evidence import EvidenceClassification, MetricSpec


def classify_effect(
    *,
    absolute_effect: float,
    confidence_interval: tuple[float, float],
    p_value: float,
    alpha: float,
    metric_spec: MetricSpec,
) -> EvidenceClassification:
    """Classify an effect using statistical and practical significance.

    Effects and intervals are transformed onto a favorable-direction scale,
    where positive is better regardless of the metric's natural direction.
    A point estimate exactly on a meaningful-effect boundary is negligible;
    it must exceed the boundary to be classified clearly positive or negative.
    """

    threshold = metric_spec.meaningful_effect
    ci_lower, ci_upper = confidence_interval
    values = (absolute_effect, ci_lower, ci_upper, p_value, alpha, threshold)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("classification inputs must be finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if threshold < 0.0:
        raise ValueError("meaningful_effect must be non-negative")
    if ci_lower > ci_upper:
        raise ValueError("confidence interval lower bound must not exceed upper bound")

    if metric_spec.higher_is_better:
        favorable_effect = absolute_effect
        favorable_ci = (ci_lower, ci_upper)
    else:
        favorable_effect = -absolute_effect
        favorable_ci = (-ci_upper, -ci_lower)

    statistically_significant = p_value < alpha
    if statistically_significant and favorable_effect > threshold:
        return EvidenceClassification.CLEARLY_POSITIVE
    if statistically_significant and favorable_effect < -threshold:
        return EvidenceClassification.CLEARLY_NEGATIVE
    if statistically_significant:
        return EvidenceClassification.NEGLIGIBLE

    if favorable_ci[0] >= -threshold and favorable_ci[1] <= threshold:
        return EvidenceClassification.NEGLIGIBLE
    return EvidenceClassification.UNCERTAIN
