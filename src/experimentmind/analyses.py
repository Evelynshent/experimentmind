"""Finite deterministic analyses available to the V2 investigation planner."""

from dataclasses import dataclass

import numpy as np

from .evidence import MetricEvidence, MetricSpec
from .scenarios import ExperimentScenario
from .statistics import analyze_binary_metric, analyze_continuous_metric
from .synthetic import CONTROL, TREATMENT


@dataclass(frozen=True)
class SegmentEvidence:
    """One metric comparison within a declared segment."""

    dimension: str
    segment: str
    metric: MetricEvidence


@dataclass(frozen=True)
class SegmentationResult:
    """Treatment effects for every value of one pre-specified dimension."""

    metric_name: str
    dimension: str
    segments: tuple[SegmentEvidence, ...]


@dataclass(frozen=True)
class RevenueDecomposition:
    """Transparent revenue identity with a descriptive conditional component.

    Revenue per converted session conditions on a post-treatment outcome. Its
    interval and classification describe observed purchasers; they are not an
    intention-to-treat causal estimate.
    """

    conversion_rate: MetricEvidence
    revenue_per_converted_session: MetricEvidence
    control_recomposed_revenue: float
    treatment_recomposed_revenue: float
    control_residual: float
    treatment_residual: float


def _metric_values(scenario: ExperimentScenario, metric_name: str) -> np.ndarray:
    values = {
        "revenue_per_session": scenario.observations.revenue,
        "conversion_rate": scenario.observations.converted,
        "shipping_cost_per_session": scenario.observations.shipping_cost,
    }
    try:
        return values[metric_name]
    except KeyError as error:
        raise ValueError(f"unsupported metric: {metric_name}") from error


def _metric_spec(scenario: ExperimentScenario, metric_name: str) -> MetricSpec:
    try:
        return next(
            spec for spec in scenario.metric_specs if spec.metric_name == metric_name
        )
    except StopIteration as error:
        raise ValueError(f"scenario does not declare metric: {metric_name}") from error


def segment_metric(
    scenario: ExperimentScenario,
    *,
    metric_name: str,
    dimension: str,
    alpha: float = 0.05,
) -> SegmentationResult:
    """Analyze one metric by a dimension declared before observing results."""

    if dimension not in scenario.prespecified_dimensions:
        raise ValueError("dimension was not pre-specified for this scenario")
    dimension_values = getattr(scenario.observations, dimension, None)
    if dimension_values is None:
        raise ValueError(f"observations do not contain dimension: {dimension}")

    values = _metric_values(scenario, metric_name)
    spec = _metric_spec(scenario, metric_name)
    results: list[SegmentEvidence] = []
    for segment in sorted(str(value) for value in np.unique(dimension_values)):
        mask = dimension_values == segment
        control = values[mask & (scenario.observations.variant == CONTROL)]
        treatment = values[mask & (scenario.observations.variant == TREATMENT)]
        if metric_name == "conversion_rate":
            metric = analyze_binary_metric(
                metric_name,
                control,
                treatment,
                metric_spec=spec,
                alpha=alpha,
            )
        else:
            metric = analyze_continuous_metric(
                metric_name,
                control,
                treatment,
                metric_spec=spec,
                alpha=alpha,
            )
        results.append(SegmentEvidence(dimension, segment, metric))
    return SegmentationResult(metric_name, dimension, tuple(results))


def decompose_revenue(
    scenario: ExperimentScenario, *, alpha: float = 0.05
) -> RevenueDecomposition:
    """Decompose revenue per session into conversion and converted-user value."""

    if not scenario.supports_revenue_decomposition:
        raise ValueError("scenario does not support revenue decomposition")
    observations = scenario.observations
    control_mask = observations.variant == CONTROL
    treatment_mask = observations.variant == TREATMENT
    conversion_spec = _metric_spec(scenario, "conversion_rate")
    conversion = analyze_binary_metric(
        "conversion_rate",
        observations.converted[control_mask],
        observations.converted[treatment_mask],
        metric_spec=conversion_spec,
        alpha=alpha,
    )

    value_spec = MetricSpec(
        "revenue_per_converted_session",
        conversion_spec.role,
        higher_is_better=True,
        meaningful_effect=0.50,
    )
    control_value = observations.revenue[control_mask & observations.converted]
    treatment_value = observations.revenue[treatment_mask & observations.converted]
    converted_value = analyze_continuous_metric(
        "revenue_per_converted_session",
        control_value,
        treatment_value,
        metric_spec=value_spec,
        alpha=alpha,
    )

    control_recomposed = conversion.control_value * converted_value.control_value
    treatment_recomposed = conversion.treatment_value * converted_value.treatment_value
    control_observed = float(np.mean(observations.revenue[control_mask]))
    treatment_observed = float(np.mean(observations.revenue[treatment_mask]))
    return RevenueDecomposition(
        conversion,
        converted_value,
        control_recomposed,
        treatment_recomposed,
        control_observed - control_recomposed,
        treatment_observed - treatment_recomposed,
    )
