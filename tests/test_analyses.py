import numpy as np
import pytest

from experimentmind.analyses import decompose_revenue, segment_metric
from experimentmind.evidence import EvidenceClassification
from experimentmind.scenarios import ScenarioName, generate_scenario
from experimentmind.synthetic import CONTROL, TREATMENT


def test_segmentation_matches_direct_group_means_and_sizes() -> None:
    scenario = generate_scenario(ScenarioName.HIDDEN_HETEROGENEITY)
    result = segment_metric(
        scenario,
        metric_name="revenue_per_session",
        dimension="user_tenure",
    )

    for segment in result.segments:
        observations = scenario.observations
        segment_mask = observations.user_tenure == segment.segment
        control = observations.revenue[segment_mask & (observations.variant == CONTROL)]
        treatment = observations.revenue[
            segment_mask & (observations.variant == TREATMENT)
        ]
        assert segment.metric.control_value == pytest.approx(np.mean(control))
        assert segment.metric.treatment_value == pytest.approx(np.mean(treatment))
        assert segment.metric.sample_size_control == len(control)
        assert segment.metric.sample_size_treatment == len(treatment)
        assert (
            segment.metric.confidence_interval[0]
            <= segment.metric.absolute_effect
            <= segment.metric.confidence_interval[1]
        )

    classifications = {segment.metric.classification for segment in result.segments}
    assert classifications == {
        EvidenceClassification.CLEARLY_POSITIVE,
        EvidenceClassification.CLEARLY_NEGATIVE,
    }


def test_segmentation_rejects_post_hoc_dimension() -> None:
    scenario = generate_scenario(ScenarioName.HIDDEN_HETEROGENEITY)
    with pytest.raises(ValueError, match="pre-specified"):
        segment_metric(
            scenario,
            metric_name="revenue_per_session",
            dimension="device",
        )


def test_revenue_decomposition_is_an_exact_arm_identity() -> None:
    scenario = generate_scenario(ScenarioName.SHIPPING_TRADEOFF)
    result = decompose_revenue(scenario)

    assert result.control_recomposed_revenue == pytest.approx(
        result.conversion_rate.control_value
        * result.revenue_per_converted_session.control_value
    )
    assert result.treatment_recomposed_revenue == pytest.approx(
        result.conversion_rate.treatment_value
        * result.revenue_per_converted_session.treatment_value
    )
    assert result.control_residual == pytest.approx(0.0, abs=1e-12)
    assert result.treatment_residual == pytest.approx(0.0, abs=1e-12)
    assert result.conversion_rate.absolute_effect > 0
    assert result.revenue_per_converted_session.absolute_effect < 0


def test_decomposition_report_labels_conditional_component_as_descriptive() -> None:
    from experimentmind.investigation import investigate
    from experimentmind.policy import recommend_after_investigation
    from experimentmind.report import render_investigation_report
    from experimentmind.statistics import analyze_scenario

    scenario = generate_scenario(ScenarioName.SHIPPING_TRADEOFF)
    evidence = analyze_scenario(scenario)
    investigation = investigate(scenario, evidence)
    report = render_investigation_report(
        evidence,
        investigation,
        (),
        recommend_after_investigation(evidence),
    )

    assert "descriptive_only" in report
    assert "post-treatment outcome" in report
    assert "not randomized causal evidence" in report
