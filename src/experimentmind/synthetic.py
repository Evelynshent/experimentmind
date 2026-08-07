"""Reproducible synthetic data for one fictional e-commerce experiment."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


CONTROL = "control"
TREATMENT = "treatment"
EXPERIMENT_NAME = "Free shipping threshold: $50 to $35"


@dataclass(frozen=True)
class Observations:
    """User-session observations from a balanced, two-arm experiment."""

    user_id: NDArray[np.int64]
    variant: NDArray[np.str_]
    converted: NDArray[np.bool_]
    revenue: NDArray[np.float64]
    shipping_cost: NDArray[np.float64]

    def __post_init__(self) -> None:
        lengths = {
            len(self.user_id),
            len(self.variant),
            len(self.converted),
            len(self.revenue),
            len(self.shipping_cost),
        }
        if len(lengths) != 1:
            raise ValueError("all observation columns must have equal length")

    def __len__(self) -> int:
        return len(self.user_id)


def generate_shipping_threshold_experiment(
    *, seed: int = 42, users_per_variant: int = 10_000
) -> Observations:
    """Generate a balanced synthetic threshold experiment.

    Lowering the threshold raises conversion, reduces order value, and makes
    free shipping both more common and more costly. The opposing conversion
    and order-value effects leave revenue per session comparatively ambiguous.
    The generator is purely fictional and deterministic for a given seed.
    """

    if users_per_variant < 2:
        raise ValueError("users_per_variant must be at least 2")

    rng = np.random.default_rng(seed)
    n = 2 * users_per_variant
    variant = np.repeat(
        np.array([CONTROL, TREATMENT]), users_per_variant
    )
    rng.shuffle(variant)
    treatment = variant == TREATMENT

    conversion_probability = np.where(treatment, 0.041, 0.032)
    converted = rng.random(n) < conversion_probability

    # Lognormal order values are positive and right-skewed. Parameters are
    # chosen by arithmetic mean and a shared log-scale standard deviation.
    order_value_mean = np.where(treatment, 49.5, 58.0)
    log_sigma = 0.42
    log_mu = np.log(order_value_mean) - 0.5 * log_sigma**2
    potential_order_value = rng.lognormal(log_mu, log_sigma)
    revenue = np.where(converted, potential_order_value, 0.0)

    # Treatment purchasers qualify more often and face a slightly higher
    # average fulfillment cost. Non-purchasers incur no shipping cost.
    free_shipping_probability = np.where(treatment, 0.84, 0.61)
    free_shipping = rng.random(n) < free_shipping_probability
    fulfillment_cost = np.maximum(
        rng.normal(loc=np.where(treatment, 5.45, 4.85), scale=0.75), 0.0
    )
    shipping_cost = np.where(converted & free_shipping, fulfillment_cost, 0.0)

    return Observations(
        user_id=np.arange(1, n + 1, dtype=np.int64),
        variant=variant,
        converted=converted,
        revenue=revenue.astype(np.float64),
        shipping_cost=shipping_cost.astype(np.float64),
    )
