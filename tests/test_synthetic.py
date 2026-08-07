import numpy as np

from experimentmind.evidence import EvidenceClassification
from experimentmind.policy import Decision, recommend
from experimentmind.statistics import analyze_experiment
from experimentmind.synthetic import (
    CONTROL,
    TREATMENT,
    generate_shipping_threshold_experiment,
)


def test_generation_is_reproducible_for_the_same_seed() -> None:
    first = generate_shipping_threshold_experiment(seed=42, users_per_variant=500)
    second = generate_shipping_threshold_experiment(seed=42, users_per_variant=500)

    for field in ("user_id", "variant", "converted", "revenue", "shipping_cost"):
        assert np.array_equal(getattr(first, field), getattr(second, field))


def test_different_seed_changes_generated_outcomes() -> None:
    first = generate_shipping_threshold_experiment(seed=42, users_per_variant=500)
    second = generate_shipping_threshold_experiment(seed=43, users_per_variant=500)

    assert not np.array_equal(first.variant, second.variant)
    assert not np.array_equal(first.converted, second.converted)


def test_observation_constraints_and_balanced_assignment() -> None:
    observations = generate_shipping_threshold_experiment(
        seed=42, users_per_variant=500
    )

    assert len(observations) == 1_000
    assert np.count_nonzero(observations.variant == CONTROL) == 500
    assert np.count_nonzero(observations.variant == TREATMENT) == 500
    assert np.all(observations.revenue >= 0)
    assert np.all(observations.shipping_cost >= 0)
    assert np.all(observations.revenue[~observations.converted] == 0)
    assert np.all(observations.shipping_cost[~observations.converted] == 0)


def test_fixed_demo_seed_encodes_a_nontrivial_tradeoff() -> None:
    evidence = analyze_experiment(
        generate_shipping_threshold_experiment(seed=42)
    )
    metrics = {metric.metric_name: metric for metric in evidence.metrics}

    assert metrics["conversion_rate"].absolute_effect > 0
    assert metrics["conversion_rate"].p_value < evidence.alpha
    assert metrics["shipping_cost_per_session"].absolute_effect > 0
    assert metrics["shipping_cost_per_session"].p_value < evidence.alpha
    assert metrics["revenue_per_session"].confidence_interval[0] < 0
    assert metrics["revenue_per_session"].confidence_interval[1] > 0
    assert (
        metrics["revenue_per_session"].classification
        is EvidenceClassification.UNCERTAIN
    )
    assert (
        metrics["conversion_rate"].classification
        is EvidenceClassification.CLEARLY_POSITIVE
    )
    assert (
        metrics["shipping_cost_per_session"].classification
        is EvidenceClassification.CLEARLY_NEGATIVE
    )
    assert recommend(evidence).decision is Decision.TRADEOFF
