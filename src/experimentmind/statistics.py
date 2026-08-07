"""Authoritative statistical calculations for a two-arm experiment.

Assumptions
-----------
Observations are independently randomized user sessions. Binary outcomes use
a two-sided, large-sample two-proportion z-test and a Newcombe confidence
interval. Continuous outcomes use a two-sided Welch t-test and Welch interval,
which do not assume equal arm variances. Intervals cover the absolute effect
(treatment mean minus control mean). These methods estimate intention-to-treat
effects because every randomized session remains in its assigned arm.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from .classification import classify_effect
from .evidence import Evidence, MetricEvidence, MetricRole, MetricSpec, MetricType
from .synthetic import CONTROL, EXPERIMENT_NAME, TREATMENT, Observations


def _validate_alpha(alpha: float) -> None:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")


def _split(
    values: NDArray[np.generic], variant: NDArray[np.str_]
) -> tuple[NDArray[np.generic], NDArray[np.generic]]:
    control = values[variant == CONTROL]
    treatment = values[variant == TREATMENT]
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("each variant must contain at least two observations")
    return control, treatment


def _relative_effect(control: float, absolute_effect: float) -> float | None:
    return None if control == 0.0 else absolute_effect / control


def analyze_binary_metric(
    metric_name: str,
    control: NDArray[np.bool_],
    treatment: NDArray[np.bool_],
    *,
    metric_spec: MetricSpec,
    alpha: float = 0.05,
) -> MetricEvidence:
    """Analyze a binary metric using standard two-proportion methods."""

    _validate_alpha(alpha)
    if metric_spec.metric_name != metric_name:
        raise ValueError("metric_spec name must match metric_name")
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("each variant must contain at least two observations")

    control_count = int(np.count_nonzero(control))
    treatment_count = int(np.count_nonzero(treatment))
    total_successes = control_count + treatment_count
    total_observations = len(control) + len(treatment)
    if total_successes in (0, total_observations):
        raise ValueError("binary metric variance is insufficient for inference")

    control_value = control_count / len(control)
    treatment_value = treatment_count / len(treatment)
    absolute_effect = treatment_value - control_value

    _, p_value = proportions_ztest(
        count=[treatment_count, control_count],
        nobs=[len(treatment), len(control)],
        alternative="two-sided",
    )
    ci_lower, ci_upper = confint_proportions_2indep(
        count1=treatment_count,
        nobs1=len(treatment),
        count2=control_count,
        nobs2=len(control),
        method="newcomb",
        compare="diff",
        alpha=alpha,
    )

    confidence_interval = (float(ci_lower), float(ci_upper))
    return MetricEvidence(
        metric_name=metric_name,
        metric_type=MetricType.BINARY,
        metric_spec=metric_spec,
        control_value=control_value,
        treatment_value=treatment_value,
        absolute_effect=absolute_effect,
        relative_effect=_relative_effect(control_value, absolute_effect),
        confidence_interval=confidence_interval,
        confidence_level=1.0 - alpha,
        p_value=float(p_value),
        sample_size_control=len(control),
        sample_size_treatment=len(treatment),
        classification=classify_effect(
            absolute_effect=absolute_effect,
            confidence_interval=confidence_interval,
            p_value=float(p_value),
            alpha=alpha,
            metric_spec=metric_spec,
        ),
    )


def analyze_continuous_metric(
    metric_name: str,
    control: NDArray[np.float64],
    treatment: NDArray[np.float64],
    *,
    metric_spec: MetricSpec,
    alpha: float = 0.05,
) -> MetricEvidence:
    """Analyze a continuous metric using Welch's unequal-variance test."""

    _validate_alpha(alpha)
    if metric_spec.metric_name != metric_name:
        raise ValueError("metric_spec name must match metric_name")
    if len(control) < 2 or len(treatment) < 2:
        raise ValueError("each variant must contain at least two observations")
    if not np.isfinite(control).all() or not np.isfinite(treatment).all():
        raise ValueError("metric values must be finite")
    if np.var(control, ddof=1) == 0.0 and np.var(treatment, ddof=1) == 0.0:
        raise ValueError("metric variance is insufficient for inference")

    control_value = float(np.mean(control))
    treatment_value = float(np.mean(treatment))
    absolute_effect = treatment_value - control_value
    test = stats.ttest_ind(treatment, control, equal_var=False)
    comparison = CompareMeans(DescrStatsW(treatment), DescrStatsW(control))
    ci_lower, ci_upper = comparison.tconfint_diff(
        alpha=alpha, alternative="two-sided", usevar="unequal"
    )

    p_value = float(test.pvalue)
    if not math.isfinite(p_value):
        raise ValueError("metric variance is insufficient for inference")

    confidence_interval = (float(ci_lower), float(ci_upper))
    return MetricEvidence(
        metric_name=metric_name,
        metric_type=MetricType.CONTINUOUS,
        metric_spec=metric_spec,
        control_value=control_value,
        treatment_value=treatment_value,
        absolute_effect=absolute_effect,
        relative_effect=_relative_effect(control_value, absolute_effect),
        confidence_interval=confidence_interval,
        confidence_level=1.0 - alpha,
        p_value=p_value,
        sample_size_control=len(control),
        sample_size_treatment=len(treatment),
        classification=classify_effect(
            absolute_effect=absolute_effect,
            confidence_interval=confidence_interval,
            p_value=p_value,
            alpha=alpha,
            metric_spec=metric_spec,
        ),
    )


def analyze_experiment(
    observations: Observations, *, alpha: float = 0.05
) -> Evidence:
    """Convert raw session observations into immutable statistical evidence."""

    _validate_alpha(alpha)
    known_variants = np.isin(observations.variant, [CONTROL, TREATMENT])
    if not known_variants.all():
        raise ValueError("variant must be either 'control' or 'treatment'")

    conversion_control, conversion_treatment = _split(
        observations.converted, observations.variant
    )
    revenue_control, revenue_treatment = _split(
        observations.revenue, observations.variant
    )
    shipping_control, shipping_treatment = _split(
        observations.shipping_cost, observations.variant
    )

    metrics = (
        analyze_continuous_metric(
            "revenue_per_session",
            revenue_control,
            revenue_treatment,
            metric_spec=MetricSpec(
                metric_name="revenue_per_session",
                role=MetricRole.PRIMARY,
                higher_is_better=True,
                meaningful_effect=0.05,
            ),
            alpha=alpha,
        ),
        analyze_binary_metric(
            "conversion_rate",
            conversion_control,
            conversion_treatment,
            metric_spec=MetricSpec(
                metric_name="conversion_rate",
                role=MetricRole.SECONDARY,
                higher_is_better=True,
                meaningful_effect=0.002,
            ),
            alpha=alpha,
        ),
        analyze_continuous_metric(
            "shipping_cost_per_session",
            shipping_control,
            shipping_treatment,
            metric_spec=MetricSpec(
                metric_name="shipping_cost_per_session",
                role=MetricRole.GUARDRAIL,
                higher_is_better=False,
                meaningful_effect=0.02,
            ),
            alpha=alpha,
        ),
    )
    return Evidence(experiment_name=EXPERIMENT_NAME, metrics=metrics, alpha=alpha)
