"""Small, pre-specified synthetic scenarios for V2 investigation."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .evidence import MetricRole, MetricSpec
from .synthetic import (
    CONTROL,
    TREATMENT,
    Observations,
    generate_shipping_threshold_experiment,
)


class ScenarioName(str, Enum):
    """The three supported V2 demonstration scenarios."""

    CLEAR_WIN = "clear_win"
    SHIPPING_TRADEOFF = "shipping_tradeoff"
    HIDDEN_HETEROGENEITY = "hidden_heterogeneity"


@dataclass(frozen=True)
class ExperimentScenario:
    """Data and analysis constraints declared before looking at results."""

    key: ScenarioName
    name: str
    hypothesis: str
    treatment_description: str
    observations: Observations
    metric_specs: tuple[MetricSpec, ...]
    prespecified_dimensions: tuple[str, ...] = ()
    supports_revenue_decomposition: bool = False


def _metric_specs(*, revenue_threshold: float = 0.05) -> tuple[MetricSpec, ...]:
    return (
        MetricSpec(
            "revenue_per_session",
            MetricRole.PRIMARY,
            higher_is_better=True,
            meaningful_effect=revenue_threshold,
        ),
        MetricSpec(
            "conversion_rate",
            MetricRole.SECONDARY,
            higher_is_better=True,
            meaningful_effect=0.002,
        ),
        MetricSpec(
            "shipping_cost_per_session",
            MetricRole.GUARDRAIL,
            higher_is_better=False,
            meaningful_effect=0.02,
        ),
    )


def _balanced_variants(rng: np.random.Generator, users_per_variant: int) -> np.ndarray:
    variant = np.repeat(np.array([CONTROL, TREATMENT]), users_per_variant)
    rng.shuffle(variant)
    return variant


def _observations_from_rates(
    *,
    rng: np.random.Generator,
    variant: np.ndarray,
    conversion_probability: np.ndarray,
    order_value_mean: np.ndarray,
    user_tenure: np.ndarray | None = None,
    shipping_probability: np.ndarray | None = None,
) -> Observations:
    n = len(variant)
    converted = rng.random(n) < conversion_probability
    log_sigma = 0.38
    log_mu = np.log(order_value_mean) - 0.5 * log_sigma**2
    potential_order_value = rng.lognormal(log_mu, log_sigma)
    revenue = np.where(converted, potential_order_value, 0.0)

    if shipping_probability is None:
        shipping_probability = np.full(n, 0.65)
    free_shipping = rng.random(n) < shipping_probability
    fulfillment_cost = np.maximum(rng.normal(5.0, 0.7, n), 0.0)
    shipping_cost = np.where(converted & free_shipping, fulfillment_cost, 0.0)

    return Observations(
        user_id=np.arange(1, n + 1, dtype=np.int64),
        variant=variant,
        converted=converted,
        revenue=revenue.astype(np.float64),
        shipping_cost=shipping_cost.astype(np.float64),
        user_tenure=user_tenure,
    )


def generate_clear_win_scenario(
    *, seed: int = 42, users_per_variant: int = 10_000
) -> ExperimentScenario:
    """Generate a checkout redesign with a clear primary-metric win."""

    if users_per_variant < 2:
        raise ValueError("users_per_variant must be at least 2")
    rng = np.random.default_rng(seed)
    variant = _balanced_variants(rng, users_per_variant)
    treatment = variant == TREATMENT
    observations = _observations_from_rates(
        rng=rng,
        variant=variant,
        conversion_probability=np.where(treatment, 0.065, 0.050),
        order_value_mean=np.where(treatment, 61.0, 60.0),
        shipping_probability=np.where(treatment, 0.48, 0.64),
    )
    return ExperimentScenario(
        key=ScenarioName.CLEAR_WIN,
        name="Checkout redesign",
        hypothesis="A simpler checkout increases completed purchases and revenue.",
        treatment_description="Replace the existing checkout with a shorter checkout flow.",
        observations=observations,
        metric_specs=_metric_specs(revenue_threshold=0.10),
    )


def generate_shipping_tradeoff_scenario(
    *, seed: int = 42, users_per_variant: int = 10_000
) -> ExperimentScenario:
    """Wrap the V1 shipping-threshold experiment in V2 metadata."""

    return ExperimentScenario(
        key=ScenarioName.SHIPPING_TRADEOFF,
        name="Free shipping threshold: $50 to $35",
        hypothesis=(
            "A lower free-shipping threshold increases conversion enough to "
            "improve revenue without unacceptable shipping cost."
        ),
        treatment_description="Lower the free-shipping threshold from $50 to $35.",
        observations=generate_shipping_threshold_experiment(
            seed=seed, users_per_variant=users_per_variant
        ),
        metric_specs=_metric_specs(),
        supports_revenue_decomposition=True,
    )


def generate_hidden_heterogeneity_scenario(
    *, seed: int = 42, users_per_variant: int = 10_000
) -> ExperimentScenario:
    """Generate a neutral aggregate effect masking tenure heterogeneity."""

    if users_per_variant < 2:
        raise ValueError("users_per_variant must be at least 2")
    rng = np.random.default_rng(seed)
    variant = _balanced_variants(rng, users_per_variant)
    treatment = variant == TREATMENT
    user_tenure = np.where(rng.random(len(variant)) < 0.40, "new", "existing")
    new_user = user_tenure == "new"

    control_probability = np.where(new_user, 0.035, 0.060)
    treatment_probability = np.where(new_user, 0.0575, 0.045)
    observations = _observations_from_rates(
        rng=rng,
        variant=variant,
        conversion_probability=np.where(
            treatment, treatment_probability, control_probability
        ),
        order_value_mean=np.full(len(variant), 58.0),
        user_tenure=user_tenure,
        shipping_probability=np.full(len(variant), 0.64),
    )
    return ExperimentScenario(
        key=ScenarioName.HIDDEN_HETEROGENEITY,
        name="Recommendation ranking change",
        hypothesis=(
            "A new ranking improves revenue, with user tenure declared in "
            "advance as a potential effect modifier."
        ),
        treatment_description="Replace the existing recommendation ranking with a new ranking.",
        observations=observations,
        metric_specs=_metric_specs(revenue_threshold=0.10),
        prespecified_dimensions=("user_tenure",),
        supports_revenue_decomposition=True,
    )


def generate_scenario(
    scenario: ScenarioName | str,
    *,
    seed: int = 42,
    users_per_variant: int = 10_000,
) -> ExperimentScenario:
    """Generate one of the finite supported scenarios."""

    key = ScenarioName(scenario)
    generators = {
        ScenarioName.CLEAR_WIN: generate_clear_win_scenario,
        ScenarioName.SHIPPING_TRADEOFF: generate_shipping_tradeoff_scenario,
        ScenarioName.HIDDEN_HETEROGENEITY: generate_hidden_heterogeneity_scenario,
    }
    return generators[key](seed=seed, users_per_variant=users_per_variant)
