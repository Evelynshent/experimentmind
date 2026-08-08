import numpy as np

from experimentmind.evidence import EvidenceClassification
from experimentmind.scenarios import (
    ScenarioName,
    generate_scenario,
)
from experimentmind.statistics import analyze_scenario


def test_all_scenarios_are_reproducible() -> None:
    for scenario_name in ScenarioName:
        first = generate_scenario(scenario_name, seed=17, users_per_variant=500)
        second = generate_scenario(scenario_name, seed=17, users_per_variant=500)
        for field in (
            "user_id",
            "variant",
            "converted",
            "revenue",
            "shipping_cost",
        ):
            assert np.array_equal(
                getattr(first.observations, field),
                getattr(second.observations, field),
            )
        if first.observations.user_tenure is not None:
            assert np.array_equal(
                first.observations.user_tenure,
                second.observations.user_tenure,
            )


def test_clear_win_has_positive_primary_without_harmed_guardrail() -> None:
    evidence = analyze_scenario(generate_scenario(ScenarioName.CLEAR_WIN))
    metrics = {metric.metric_name: metric for metric in evidence.metrics}

    assert (
        metrics["revenue_per_session"].classification
        is EvidenceClassification.CLEARLY_POSITIVE
    )
    assert (
        metrics["shipping_cost_per_session"].classification
        is not EvidenceClassification.CLEARLY_NEGATIVE
    )


def test_hidden_heterogeneity_is_neutral_at_top_line() -> None:
    evidence = analyze_scenario(generate_scenario(ScenarioName.HIDDEN_HETEROGENEITY))
    primary = next(
        metric
        for metric in evidence.metrics
        if metric.metric_name == "revenue_per_session"
    )

    assert primary.classification is EvidenceClassification.UNCERTAIN
    assert primary.confidence_interval[0] < 0 < primary.confidence_interval[1]
